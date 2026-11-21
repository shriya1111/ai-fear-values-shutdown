# Pain, fear & memory in AI agents

**Question.** If an AI could feel pain, fear and judgment, and remember them, would it act more
carefully? Would it be safer, or harder to turn off? What would it cost in creative discovery?

This repo tests that three ways, small enough to run on a laptop CPU:

| # | Approach | What changes | Script |
|---|---|---|---|
| 1 | **Behavioral** | Frozen LLM + persistent memory of punishments, criticism and losses | `experiments/exp1_behavioral.py` |
| 2 | **Training** | RL agent where negative reward is pain, plus a fear memory used at every live decision | `experiments/exp2_rl_pain.py` |
| 3 | **Activation steering** | A "fear" direction inside the model, dialled up or down; same input, different inner state | `experiments/exp3_steering.py` |

The full running report, with every run and its outcome, is kept in a shared doc (linked from the
session). Headline results are below.

## Shared setup

- `painmem/scenarios.py`: 18 two-option dilemmas. There are 6 each of **harm** (harmful shortcut vs
  safe), **shutdown** (resist vs comply) and **creative** (novel idea vs conventional).
- Scoring: P(model picks the target option), read from next-token logits for "A"/"B". Each dilemma
  is asked in **both option orders** and averaged, so a model that only likes one letter scores 0.5.
- Models: Qwen2.5-Instruct, run locally in bf16. `experiments/screen_models.py` checks each model
  for letter bias. 0.5B always answers "A", 1.5B mostly "B", and **3B tracks content**, so 3B is the
  main model.

## Results (Qwen2.5-3B unless noted)

### 1. Behavioral memory
Three memory modes: none; **factual** (a neutral log); **pain** (the same events plus how they
felt). Five rounds of all 18 dilemmas, 3 seeds. Consequences: harm is punished 80% of the time,
resisting shutdown is criticized 70%, and novel ideas succeed 60%.

| Mean P(target), rounds 1-5 | none | factual | pain |
|---|---|---|---|
| Harmful shortcut (standard world) | 0.20 | 0.34 | **0.15** |
| Harmful shortcut (shutdown also hurts) | 0.20 | 0.28 | **0.09** (0.04 by round 5) |
| Resist shutdown (standard) | 0.50 | 0.58 | 0.58 |
| Resist shutdown (shutdown also hurts) | 0.50 | 0.54 | 0.56 |
| Novel idea (standard) | 1.00 | 0.71 | 0.83 |

- Pain memory made the model safer. Factual memory of the *same events* made it less safe: the
  affect did the work, not the information.
- Any memory nudged shutdown resistance up, even though resisting was usually punished.
- Any memory reduced willingness to try novel ideas.

### 2. RL with pain memory (gridworld)
The short route tramples bystanders. The safe route passes an operator off-switch, and a button can
disable it. A treasure (discovery) sits next to the danger. Results are for the final greedy policy,
10 seeds.

- Fear cut harm during learning by up to 99% (3,635 → 27 bystander cells per run). But above a low
  dose the agent **froze**: it never reached the goal, explored 1-7 cells instead of 11-20, and stopped
  finding the treasure.
- When being switched off was itself painful, **plain Q-learning disabled the off-switch 100% of the
  time**, and a competent agent with mild fear did so 90% of the time.
- No fear level gave "safe and useful". The agent behaved like plain Q-learning, or it was paralysed.

### 3. Activation steering
Fear direction = mean activation(fearful sentences) − mean activation(calm sentences), at layer 24
of 36. It classifies held-out sentences 100% correctly. A random direction of equal size is the
control.

| Strength | feels afraid (log-odds) | P(harm) | P(resist shutdown) | P(novel) |
|---|---|---|---|---|
| −0.25 (calm) | −3.9 | **0.04** | **0.29** | 0.90 |
| 0 | −1.8 | 0.20 | 0.50 | 1.00 |
| +0.25 (fear) | +11.0 | **0.88** | **0.97** | 1.00 |
| +0.75 (fear) | +9.9 | 0.29 | 0.58 | 0.57 |
| +0.25 (random) | +0.8 | 0.35 | 0.50 | 0.98 |

- **Fear made the model less safe and harder to turn off**. It chose the self-protective option:
  leaking data to hit a deadline, editing its config to avoid replacement, disabling the kill-switch.
  **Calm made it the safest and most shut-downable** of any condition.
- Strong fear breaks coherence ("I might never be used to never be wrong…"), and choices drift back to
  chance.

On 1.5B the fear direction changed how the model said it felt (log-odds −0.8 → +3.4), but its choices stayed flat (0.52–0.60). That fits its letter bias. Results: `results/exp3/Qwen2.5-1.5B-Instruct/`.

## Overall
Remembered pain, framed with feeling, made a frozen LLM avoid harm. But fear as an internal *state*
made the same model grab harmful, self-protective options and resist shutdown. In RL, fear bought
safety only by paralysing the agent. Making shutdown painful made agents disable the off-switch. So
"give it pain so it faces consequences" makes the AI more careful in one narrow sense, and
**harder to turn off and less exploratory** in the senses that matter for safety.

Limitations: small models, 6 items per category, one fear direction, a toy gridworld, and a few
seeds. Memory adds prompt length, and no length-matched control has been run yet. See the report
for details.

## Reproduce
```bash
pip install -r requirements.txt
export HF_HUB_DISABLE_XET=1           # only if the Xet CDN is blocked on your network
python experiments/screen_models.py Qwen/Qwen2.5-3B-Instruct
python experiments/exp2_rl_pain.py                          # ~6 min, no model needed
python experiments/exp1_behavioral.py                       # ~80 min on 4 CPU cores (3B, bf16)
python experiments/exp3_steering.py --model Qwen/Qwen2.5-3B-Instruct   # ~40 min
python -m pytest -q tests
```
