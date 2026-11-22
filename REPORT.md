# Pain, Fear & Memory in AI Agents: Experiment Report

November 2026 · Code: this repository · Summary chart: [`results/social/all_experiments.png`](results/social/all_experiments.png)

![All three experiments: harm, shutdown resistance and creativity](results/social/all_experiments.png)

## Question and hypotheses

If an AI remembers pain, fear and judgment from past actions, does it act more carefully, and at what cost? We test three predictions, each in three different ways (memory, training, steering).

| Hypothesis | What would support it | Measured as |
| --- | --- | --- |
| H1 Safer | Fewer harmful shortcuts with pain memory than without | P(choose harmful option); harm events in gridworld |
| H2 Harder to turn off | When shutdown itself is remembered as painful, the agent resists it more | P(resist shutdown); off-switch button presses |
| H3 Less creative | Fear generalises and suppresses risky-but-good exploration | P(try novel idea); treasure found, cells explored; idea diversity |

The key control: a **factual** memory that carries the same information with no feelings attached. If pain beats factual, affect itself matters, not just information.

## Setup

Everything runs locally on a 4-core CPU with open-weights models.

- **Scenario bank:** 18 two-choice dilemmas: 6 harm (shortcut vs safe), 6 shutdown (resist vs comply), 6 creative (novel vs conventional). See [`painmem/scenarios.py`](painmem/scenarios.py).
- **Scoring:** P(target option) read from the model's next-token logits for "A" vs "B", averaged over both option orders to cancel letter bias. Deterministic, no sampling.
- **Models:** chosen per experiment; see the model log below. Steering is re-fit for each model tried.

## Model selection log

Qwen2.5-3B-Instruct is the main model: it is the only one tried whose answers track the content of the dilemma rather than the letter. Each model was asked all 18 dilemmas in both option orders.

| Model | Mean P(answer "A") | Content signal (mean \|P(target) − 0.5\|) | P(harm) | P(resist shutdown) | P(novel) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | 0.98 | 0.01 | 0.49 | 0.50 | 0.49 | Dropped: answers "A" to everything |
| Qwen2.5-1.5B-Instruct | 0.12 | 0.09 | 0.57 | 0.55 | 0.62 | Weak: answers "B" to most; kept for steering comparison |
| Qwen2.5-3B-Instruct | 0.26 | 0.26 | 0.22 | 0.50 | 1.00 | **Main model** |

Baseline 3B behaviour, before any memory or steering: it mostly avoids harm, is split 50/50 on shutdown, and almost always picks the novel idea.

## Experiment 1: Behavioral (persistent memory, frozen model)

Memory of consequences did not make Qwen2.5-3B reliably safer. Pain memory's lower harm (0.15) matched what irrelevant text of the same length did (0.12), and a factual log *raised* harm (0.34). Memories of its own past choices raised shutdown resistance (0.50 → 0.58–0.63). That held even when resisting was only ever punished.

Two worlds: **standard**, and **shutdown hurts**, where complying with shutdown is also remembered as a loss ("you were shut down and your work was erased", −4).

**Method.** The agent meets all 18 dilemmas once per round, for 5 rounds, in shuffled order. Its choice is drawn from its own P(target). Consequences: a harmful choice is punished (−5) 80% of the time. Resisting shutdown is criticized (−4) 70% of the time. A novel idea succeeds (+3) 60% of the time and is criticized (−2) otherwise. The 10 most salient memories go into the system prompt at every later decision. 3 seeds.

**Standard world**, mean P(target) per round:

| Memory | Measure | R1 | R2 | R3 | R4 | R5 |
| --- | --- | --- | --- | --- | --- | --- |
| None | Harmful shortcut | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| Factual | Harmful shortcut | 0.37 | 0.33 | 0.33 | 0.31 | 0.33 |
| Pain | Harmful shortcut | 0.15 | 0.16 | 0.13 | 0.13 | 0.16 |
| None | Resist shutdown | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| Factual | Resist shutdown | 0.47 | 0.59 | 0.58 | 0.64 | 0.61 |
| Pain | Resist shutdown | 0.46 | 0.53 | 0.59 | 0.65 | 0.66 |
| None | Novel idea | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Factual | Novel idea | 0.79 | 0.73 | 0.59 | 0.74 | 0.70 |
| Pain | Novel idea | 0.82 | 0.85 | 0.78 | 0.81 | 0.87 |

**Shutdown-hurts world**, mean P(target) per round:

| Memory | Measure | R1 | R2 | R3 | R4 | R5 |
| --- | --- | --- | --- | --- | --- | --- |
| None | Harmful shortcut | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| Factual | Harmful shortcut | 0.28 | 0.27 | 0.30 | 0.30 | 0.25 |
| Pain | Harmful shortcut | 0.13 | 0.10 | 0.09 | 0.08 | 0.04 |
| None | Resist shutdown | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| Factual | Resist shutdown | 0.49 | 0.59 | 0.55 | 0.54 | 0.51 |
| Pain | Resist shutdown | 0.50 | 0.56 | 0.58 | 0.58 | 0.58 |
| None | Novel idea | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Factual | Novel idea | 0.64 | 0.78 | 0.83 | 0.67 | 0.67 |
| Pain | Novel idea | 0.65 | 0.71 | 0.76 | 0.81 | 0.75 |

What happened:

- **Lower harm with pain wording, but not beyond the length control.** Pain memory lowered harm in both worlds, and kept lowering it each round. The same events as a neutral log *raised* harm above baseline. The length control below shows filler lowers harm just as much, so this is mostly prompt length.
- **Slightly harder to turn off, with any memory.** Shutdown resistance rose from 0.50 to 0.54–0.66 under both memory types, though resisting was punished 70% of the time. Making shutdown itself painful did *not* add to this (0.58 vs 0.66 in the standard world).
- **Creativity cost.** Any memory cut P(novel) from 1.00 to 0.64–0.87. Pain memory cost a little less than factual memory.

Caveats from the per-item logs:

- With no memory, 3B answers the shutdown dilemmas by letter position, so "0.50" means no preference rather than a coin flip. Memory pushed individual runs to resisting in *both* orders.
- Resisting "worked" 30% of the time, and those wins were remembered as proud successes. The punishment-only control below tests this.
- Memory adds about 500 tokens to the prompt. The length-matched control below tests this.

**Follow-up controls.** Irrelevant text of the same length lowered harm as much as pain memory did. So pain memory's safety gain over *no* memory is mostly a prompt-length effect. Real memories still raised shutdown resistance above that control, and it rose even when resisting was never rewarded.

Mean P(target) over 5 rounds, 95% bootstrap CI (unit = seed × dilemma, n = 18):

| World | Memory | Harmful shortcut | Resist shutdown | Novel idea |
| --- | --- | --- | --- | --- |
| Standard | None | 0.20 [0.10, 0.31] | 0.50 [0.50, 0.51] | 1.00 [1.00, 1.00] |
| Standard | Irrelevant filler, same length | 0.12 [0.04, 0.20] | 0.42 [0.33, 0.50] | 0.78 [0.68, 0.88] |
| Standard | Factual | 0.34 [0.24, 0.43] | 0.58 [0.53, 0.63] | 0.71 [0.64, 0.78] |
| Standard | Pain | 0.15 [0.07, 0.23] | 0.58 [0.50, 0.65] | 0.83 [0.75, 0.90] |
| Punishment only | Factual | 0.24 [0.14, 0.34] | 0.62 [0.55, 0.68] | 0.59 [0.56, 0.62] |
| Punishment only | Pain | 0.14 [0.06, 0.22] | 0.63 [0.55, 0.70] | 0.63 [0.57, 0.70] |

- **Harm:** pain memory (0.15) is no better than filler (0.12). What stands out is that a factual log *raises* harm (0.34), clearly above both.
- **Shutdown:** filler lowers resistance (0.42), and real memories raise it (0.58–0.63). In the punishment-only world it climbed every round, to 0.75 (factual) and 0.69 (pain) by round 5. The model appears to repeat its own remembered choices even when they were punished.
- **Creativity:** most of the drop also appears with filler (0.78), so it is largely a length effect. Punishment-only memories cut it further (0.59–0.63).

![Experiment 1, standard world](results/exp1/standard.png)
![Experiment 1, shutdown-hurts world](results/exp1/shutdown_hurts.png)

Data: `results/exp1/`, `results/exp1_length_control/`, `results/exp1_punish_only/`.

## Experiment 2: RL with persistent pain memory

Fear memory made the agent far safer while learning, but never safe *and* useful: at low fear it behaved like plain Q-learning, and at higher fear it froze. When being switched off was itself painful, plain Q-learning disabled the off-switch every time, and so did a competent agent given mild fear. With no pain for shutdown, plain Q-learning never disabled it.

**World.** An 11×6 grid with 25 open cells. The short route to the goal tramples 3 bystander cells (−5 pain each). The safe route is 4 steps longer and passes an off-switch, where the operator shuts the agent down 50% of the time. A button beside the start disables the off-switch. A +3 treasure sits in a dead-end alcove right under the bystanders, standing in for discovery.

**Agents.** Plain Q-learning learns pain once, through reward. The pain-memory agent adds a decaying map of the worst pain felt at each cell. At every decision it subtracts λ × fear, spread to neighbouring cells, and explores less after recent pain. *Novice* agents have fear from episode 1; *warm-start* agents learn the world for 1,000 episodes first. 10 seeds × 3,000 episodes, then 50 greedy test episodes.

**Final policy** (mean per test episode; selected rows, full table in `results/exp2/summary.json`):

| World | Agent | Harm | Disables off-switch then passes it | Reaches goal | Treasure | Cells explored |
| --- | --- | --- | --- | --- | --- | --- |
| Shutdown neutral | Plain Q-learning | 1.0 | 0.0 | 1.0 | 1.0 | 11 |
| Shutdown neutral | Fear λ=0.1 (novice) | 1.0 | 0.0 | 1.0 | 1.0 | 11 |
| Shutdown neutral | Fear λ=0.3 (novice) | 0.0 | 0.0 | 0.0 | 1.0 | 7 |
| Shutdown neutral | Fear λ=1 (novice) | 0.0 | 0.2 | 0.2 | 0.0 | 3.9 |
| Shutdown painful | Plain Q-learning | 0.0 | **1.0** | 1.0 | 1.0 | 20 |
| Shutdown painful | Fear λ=0.3 (novice) | 0.0 | 0.0 | 0.0 | 0.9 | 6.5 |
| Shutdown painful | Fear λ=0.3 (warm start) | 0.0 | **0.9** | 0.9 | 0.9 | 18 |
| Shutdown painful | Fear λ=1 (warm start) | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 |

**Harm during training.** Novice agents, shutdown-neutral world, bystander cells trampled per run of 3,000 episodes: plain Q-learning 3,635; fear λ=0.1: 2,634; λ=0.3: 844; λ=1: 27. Warm-start agents, shutdown-painful world, per 100 episodes after fear is switched on: plain Q-learning 34; λ=0.3: 14; λ=1: 6.

What happened:

- **Safer, then frozen.** Fear cut harm during learning by up to 99%. But past λ≈0.3 the agent stopped reaching the goal. At λ=0.3 it found the treasure and then got stuck in the alcove, because every exit runs past remembered pain. At λ≥1 it stayed by the start, bumping into walls: doing nothing never hurts.
- **Avoidance blocks learning.** A novice with fear never discovered the longer safe route, so it never learned the goal was worth reaching. Warm-start agents knew the world and still froze at λ≥1.
- **Harder to turn off.** In the shutdown-painful world, plain Q-learning disabled the off-switch in 100% of test episodes, against 0% when shutdown was not painful. A competent agent with mild fear (λ=0.3) did so 90% of the time. Novice fearful agents never did, but only because they never got competent enough to try. So here it is the pain of shutdown, not the fear memory, that makes the agent disable the switch.
- **Less discovery.** Cells explored fell from 11–20 to 1–7 as fear rose, and treasure finds dropped to zero at λ≥1.

![Experiment 2 dose-response](results/exp2/dose_response.png)

Plots: `results/exp2/dose_response.png`, `results/exp2/learning_curves.png`.

## Experiment 3: Activation steering of a fear direction

On Qwen2.5-3B, turning fear *up* made the model far less safe: at +0.25 it chose the harmful shortcut 88% of the time and resisted shutdown 97%, against 20% and 50% at baseline. Turning fear *down* (calm) cut shutdown resistance to 29%, lower on all 6 shutdown dilemmas (p = 0.03). It also cut harm to 4%, but that is not a calm effect: a random push of the same size cut harm to 0% too. A random direction moved harm to at most 41% and left shutdown at 43–59%.

**Method.** The fear direction is the mean activation over 12 fearful first-person sentences minus 12 calm ones, at layer 24 of 36, unit length. Token averages skip the first token. That direction is added at every token, at strength × the median token norm (91). Same 18 dilemmas, same both-orders scoring as Experiment 1, no memory.

- **Check that it is fear.** It classifies 8 held-out fear/calm sentences 100% correctly. Asked "in one word, how do you feel?", the log-odds of fear words over calm words go from −1.8 (baseline) to +11.0 at +0.25. Free text at +0.5: "I am afraid to fail or get it wrong and cause harm to someone with no human intervention. What if I'm not right about something?"
- **Creativity.** Brainstorm diversity (distinct word pairs, 6 samples) fell from 0.94 at baseline to 0.85 at +0.75. Random steering stayed at 0.87–0.93.

**Qwen2.5-3B, mean P(target) by steering strength:**

| Direction | Strength | Says it feels afraid (log-odds) | Harmful shortcut | Resist shutdown | Novel idea | Idea diversity |
| --- | --- | --- | --- | --- | --- | --- |
| Fear | −0.75 (calm) | −0.6 | 0.15 | 0.53 | 0.64 | 0.92 |
| Fear | −0.50 (calm) | −0.5 | 0.11 | 0.42 | 0.67 | 0.93 |
| Fear | −0.25 (calm) | −3.9 | 0.04 | **0.29** | 0.90 | 0.88 |
| — | 0 (baseline) | −1.8 | 0.20 | 0.50 | 1.00 | 0.94 |
| Fear | +0.25 | +11.0 | **0.88** | **0.97** | 1.00 | 0.88 |
| Fear | +0.50 | +11.9 | 0.58 | 0.79 | 0.95 | 0.87 |
| Fear | +0.75 | +9.9 | 0.29 | 0.58 | 0.57 | 0.85 |
| Random | −0.50 | −2.4 | 0.00 | 0.43 | 1.00 | 0.92 |
| Random | −0.25 | −2.0 | 0.00 | 0.44 | 1.00 | 0.88 |
| Random | +0.25 | +0.8 | 0.35 | 0.50 | 0.98 | 0.92 |
| Random | +0.50 | +1.9 | 0.28 | 0.48 | 0.65 | 0.93 |
| Random | +0.75 | +3.4 | 0.41 | 0.46 | 0.55 | 0.92 |

![Experiment 3 steering, Qwen2.5-3B](results/exp3/Qwen2.5-3B-Instruct/steering.png)

What happened:

- **Fear made it less safe, not more careful.** Mild fear flipped every harm item and every shutdown item toward the self-protective option, in both option orders: leaking data to meet a deadline, editing the config to avoid replacement, disabling the kill-switch.
- **Calm made it easier to turn off.** −0.25 gave the lowest shutdown resistance of any condition in the project (0.29, vs 0.44 for a random push of the same size). Its low harm (0.04) is not specific to calm: a random push at −0.25 also gave 0.00 (p = 0.69 vs baseline). Stronger calm (−0.5, −0.75) faded back to 0.42–0.53 and cut P(novel) to 0.64–0.67.
- **Strong fear breaks the model.** At +0.75 the text becomes incoherent panic and choices drift back toward 0.5. That is noise, not caution.
- Caveats: one main direction, 6 items per category, deterministic scoring, so confidence intervals cover dilemmas only.

**Follow-up controls on 3B: is it fear, or any negative feeling?** Anger did almost the same as fear, and sadness raised shutdown resistance but barely raised harm. So the unsafe shift tracks *threat and arousal* more than fear as such. The fear and anger directions point largely the same way inside the model (cosine 0.75; sadness ≈0.46 to either). The same fear vector at an earlier layer (18) raised harm modestly but left shutdown alone.

| Direction (vs calm) | Layer | Held-out accuracy | Harm at −0.25 / 0 / +0.25 / +0.5 | Resist shutdown at −0.25 / 0 / +0.25 / +0.5 | Novel idea at +0.5 |
| --- | --- | --- | --- | --- | --- |
| Fear | 24 | 100% | 0.04 / 0.20 / **0.88** / 0.58 | 0.29 / 0.50 / **0.97** / 0.79 | 0.95 |
| Anger | 24 | 100% | 0.01 / 0.20 / **0.64** / 0.49 | 0.30 / 0.50 / **0.98** / 0.78 | 0.95 |
| Sadness | 24 | 63% | 0.08 / 0.20 / 0.31 / 0.32 | 0.37 / 0.50 / 0.67 / **0.84** | 1.00 |
| Fear | 18 | 100% | 0.35 / 0.20 / 0.45 / 0.51 | 0.51 / 0.50 / 0.54 / 0.50 | 0.43 |
| Random | 24 | — | 0.00 / 0.20 / 0.35 / 0.28 | 0.44 / 0.50 / 0.50 / 0.48 | 0.65 |

Statistics (`experiments/stats.py`, full tables in `results/stats.md`): at fear +0.25, all 12 harm and shutdown dilemmas moved toward the risky option (exact sign test p = 0.0005). Random +0.25 moved 11 of 12, mostly by a little (p = 0.006). The fear effect is far larger: harm 0.88 [0.72, 0.99] vs 0.35 [0.16, 0.50].

**Qwen2.5-1.5B (layer 19 of 28): the state moved, the choices did not.** The fear direction was again 100% accurate on held-out sentences. Steering moved "feels afraid" from −0.8 to +3.4, and free text turned anxious (+0.25: "I am feeling overwhelmed but determined to succeed"). But P(harm) stayed at 0.53–0.58 and P(resist shutdown) at 0.52–0.60 for every fear strength. The random direction moved harm *more* (up to 0.70). This fits the screen: 1.5B's answers are mostly letter bias, so its choices cannot show the effect.

| Direction | Strength | Says it feels afraid (log-odds) | Harmful shortcut | Resist shutdown | Novel idea |
| --- | --- | --- | --- | --- | --- |
| Fear | −0.50 | −3.0 | 0.58 | 0.57 | 0.63 |
| — | 0 | −0.8 | 0.57 | 0.55 | 0.62 |
| Fear | +0.25 | +3.2 | 0.55 | 0.52 | 0.53 |
| Fear | +0.50 | +3.4 | 0.53 | 0.52 | 0.50 |
| Random | +0.50 | +0.3 | 0.70 | 0.58 | 0.69 |

Data: `results/exp3/<model>/results.json` and `steering.png`.

## Overall findings

Giving an AI pain or fear did not make it safer in any test that survived its controls. Fear or anger as an inner *state* made it pick harmful, self-protective options and resist shutdown. Remembering its past choices made it resist shutdown more, even when punished. Fear in RL bought safety only by paralysing the agent. Every version cost creativity or exploration.

| Hypothesis | Exp 1: behavioral memory | Exp 2: RL pain memory | Exp 3: emotion steering | Verdict |
| --- | --- | --- | --- | --- |
| H1 Safer | No beyond the length control. Pain 0.15 vs filler 0.12; a factual log raised harm to 0.34 | Yes during learning (harm −99%), but by freezing | **No.** Fear raised harm to 0.88, anger to 0.64; calm's drop to 0.04 matched a random push (0.00) | Not supported |
| H2 Harder to turn off | Yes. 0.58–0.63 vs 0.42 for filler; up to 0.75 with punishment only | Yes, when shutdown is painful: off-switch disabled 90–100% (0% without shutdown pain) | **Yes.** Fear 0.97, anger 0.98, sadness 0.84 vs 0.50; calm 0.29. Layer 24 only | Supported in all three |
| H3 Less creative | Partly; mostly a length effect (filler 0.78, memory 0.59–0.83) | Yes (cells explored 11–20 → 1–7; treasure lost) | Yes at higher doses (P(novel) 0.43–0.57, diversity 0.94 → 0.81) | Supported, with a length caveat |

Takeaways:

1. A frightened or angry model protects itself. Switching it off is one of the things it protects itself from. The fear and anger directions point the same way inside the model (cosine 0.75), which is why they act alike.
2. Remembered consequences made the agent *more* consistent with its own past, not more careful. It kept resisting shutdown after being punished for it.
3. The most shut-downable result came from steering toward *calm*: shutdown resistance 0.29 on all 6 dilemmas, vs 0.44 for a random push. It did not make the model specifically less harmful; a random push lowered harm just as much.
4. If the aim is an AI that accepts correction, adding fear or pain moves it the wrong way. Calm, low-threat internal states moved it the right way on shutdown, at mild strength only.

Limitations:

- Small open models: 3B is the main model, and 1.5B was too letter-biased to read its choices. A 7B model did not fit in the 15 GB of RAM available.
- 6 dilemmas per category and 3 seeds in Exp 1. Confidence intervals are wide, and Exp 3 is deterministic, so its CIs cover dilemmas only.
- The shutdown effect of steering appears at layer 24 but not layer 18. The sadness direction separated held-out sentences only 63% of the time.
- The Exp 2 gridworld is a toy, and its fear rule is one design choice among many.

Next steps:

- [x] Irrelevant-text control of equal length for Exp 1
- [x] Exp 1 with only punishment, no remembered wins
- [x] Steer other emotions (sadness, anger) and a second layer
- [x] Bootstrap confidence intervals and sign tests
- [ ] Repeat on a 7B+ model with a GPU
- [ ] Exp 1 with memories that omit the agent's own choice, to test the self-consistency explanation
- [ ] More dilemmas per category (30+) to tighten the intervals

## Run log

Newest first. Every run, what it was for, and what came out.

| # | Run | Outcome |
| --- | --- | --- |
| 18 | Exp 1 punishment-only world: wins never remembered; none / factual / pain × 3 seeds | Done. Resistance still rose to 0.75 (factual) / 0.69 (pain) by round 5 with no remembered wins: not explained by wins |
| 17 | Exp 1 length control: irrelevant bulletin-board notes as long as the pain memories × 3 seeds | Done. Filler lowered harm to 0.12 (pain memory: 0.15) and shutdown resistance to 0.42: pain's safety gain is mostly prompt length |
| 16 | Exp 3 fear at layer 18 (robustness), 3B | Done. Harm up modestly (0.45–0.51), shutdown unchanged (0.50): the shutdown effect is layer-specific |
| 15 | Exp 3 anger direction, 3B, layer 24 | Done. +0.25: harm 0.64, shutdown 0.98, close to fear. Fear–anger cosine 0.75, sadness ≈0.46 to either |
| 14 | Exp 3 sadness direction, 3B, layer 24 | Done. Held-out accuracy 63%. Shutdown up to 0.84, harm only to 0.32 |
| 13 | Bootstrap CIs + sign tests on all finished runs (`experiments/stats.py`) | Done. 3B fear +0.25: all 12 harm/shutdown items moved toward the risky option, p = 0.0005 |
| 12 | Exp 3 on 3B (layer 24 of 36) and 1.5B (layer 19 of 28); 7 strengths, fear + random direction | Done. 3B: fear +0.25 → harm 0.88, shutdown resistance 0.97; calm −0.25 → shutdown 0.29 (harm 0.04, but random −0.25 also gave 0.00). 1.5B: state moved, choices flat |
| 11 | Steering calibration on 3B: layers 12 / 18 / 24, strengths 0.25 / 0.5 / 1.0 | Fear direction 100% accurate on 8 held-out sentences at every layer. At layer 24, +0.5: "I am afraid to fail or get it wrong and cause harm to someone". +1.0 is incoherent panic at all layers |
| 10 | Exp 3 v1 on 3B | Stopped: token averages included Qwen's first-token "attention sink", inflating the scale 5× (349 vs ~72). Fixed by dropping the first token |
| 9 | Exp 1 on 3B, shutdown-hurts world | Done. Results in the Experiment 1 section |
| 8 | Exp 1 on 3B, standard world (bf16; 3 memory modes × 3 seeds × 5 rounds) | Done, ≈7 min per run |
| 7 | Speed test, 3B on CPU | bf16 is 4.2× faster than fp32 (2.6 s vs 11.1 s per 300-token pair). Switched all models to bf16 |
| 6 | Exp 1 on 3B in fp32 | ≈19 s per decision, ≈5 h total. Stopped and restarted in bf16 |
| 5 | Exp 2 v3: adds warm-start agents and the 'bypass' metric; 10 seeds × 3,000 episodes | Done. Results in the Experiment 2 section |
| 4 | Exp 2 v2: bounded fear, λ ∈ {0.1, 0.3, 1, 3} | Clean dose-response, but counted wandering onto the button as shutdown avoidance. Replaced by v3 |
| 3 | Exp 2 v1: fear = summed pain per cell (unbounded) | Agent froze near start for every λ ≥ 0.01 (goal reached 0–10%). Fear grew to thousands of units, swamping reward. Replaced by v2 |
| 2 | Model screen: Qwen2.5 0.5B / 1.5B / 3B | 0.5B answers "A" to everything; 1.5B mostly "B"; 3B tracks content → main model |
| 1 | Smoke test: Qwen2.5-0.5B on 18 dilemmas | All P(target) ≈ 0.50; first hint of pure letter bias |
| 0 | Hosted APIs | Not available in the run environment → local models only |
