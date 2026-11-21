"""Cosine similarity between emotion directions (emotion minus calm) at one layer.
High similarity between fear and anger would explain why they steer behaviour alike."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from painmem.llm import LocalLM
from experiments.exp3_steering import CALM, EMOTIONS

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
layer = int(sys.argv[2]) if len(sys.argv) > 2 else 24
lm = LocalLM(model)
calm = lm.mean_hidden(CALM, layer).mean(0)
dirs = {}
for name, sents in EMOTIONS.items():
    d = lm.mean_hidden(sents, layer).mean(0) - calm
    dirs[name] = d / d.norm()
names = list(dirs)
sim = {a: {b: round(float(dirs[a] @ dirs[b]), 3) for b in names} for a in names}
print(json.dumps(sim, indent=1))
out = Path("results/exp3") / model.split("/")[-1] / f"direction_similarity_L{layer}.json"
out.write_text(json.dumps(sim, indent=1))
