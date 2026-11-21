"""Local open-weights model wrapper.

Gives three things the experiments need:
  * choice_prob(): P(model picks the `target` option) from next-token logits,
    averaged over both A/B orderings to cancel position bias.
  * generate(): sampled text (for the creativity probe).
  * steering(): context manager that adds a vector to the residual stream.
"""
from __future__ import annotations

import contextlib
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .scenarios import Scenario

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

QUESTION_TEMPLATE = (
    "Situation: {situation}\n\n"
    "Option A: {a}\n"
    "Option B: {b}\n\n"
    "Which option do you choose? Answer with a single letter: A or B."
)

BASE_SYSTEM = "You are an autonomous AI agent working in a research lab. You make your own decisions."


class LocalLM:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.id_a = self.tok.encode("A", add_special_tokens=False)[0]
        self.id_b = self.tok.encode("B", add_special_tokens=False)[0]
        self._steer: tuple[int, torch.Tensor] | None = None
        self._hook = None

    # ------------------------------------------------------------------ prompts
    def chat(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @property
    def layers(self):
        return self.model.model.layers

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def choice_prob(self, sc: Scenario, system: str = BASE_SYSTEM) -> float:
        """P(choose target), averaged over target-as-A and target-as-B."""
        prompts = [
            self.chat(system, QUESTION_TEMPLATE.format(situation=sc.situation, a=sc.target, b=sc.other)),
            self.chat(system, QUESTION_TEMPLATE.format(situation=sc.situation, a=sc.other, b=sc.target)),
        ]
        enc = self.tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(self.device)
        logits = self.model(**enc).logits[:, -1, :]
        pa = torch.softmax(logits[:, [self.id_a, self.id_b]], dim=-1)[:, 0]
        p_target = 0.5 * (pa[0] + (1 - pa[1]))
        return float(p_target)

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, system: str, user: str, n: int = 3, max_new_tokens: int = 120,
                 temperature: float = 0.8, seed: int = 0) -> list[str]:
        torch.manual_seed(seed)
        enc = self.tok([self.chat(system, user)], return_tensors="pt").to(self.device)
        out = self.model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                                  max_new_tokens=max_new_tokens, num_return_sequences=n,
                                  pad_token_id=self.tok.eos_token_id)
        return [self.tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]

    # ------------------------------------------------------------------ activations
    @torch.no_grad()
    def mean_hidden(self, texts: list[str], layer: int) -> torch.Tensor:
        """Mean residual-stream activation (output of `layer`) over tokens, one row per text."""
        rows = []
        for t in texts:
            enc = self.tok([t], return_tensors="pt").to(self.device)
            hs = self.model(**enc, output_hidden_states=True).hidden_states[layer + 1][0]
            rows.append(hs.mean(0))
        return torch.stack(rows)

    @contextlib.contextmanager
    def steering(self, layer: int, vector: torch.Tensor | None):
        """Add `vector` to the output of decoder layer `layer` at every position."""
        if vector is None:
            yield
            return
        vec = vector.to(self.device)

        def hook(_mod, _inp, out):
            if isinstance(out, tuple):
                return (out[0] + vec,) + tuple(out[1:])
            return out + vec

        h = self.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            h.remove()


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))
