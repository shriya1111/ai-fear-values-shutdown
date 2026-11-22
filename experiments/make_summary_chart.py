"""One shareable image: all three experiments x three questions (harm, shutdown, creativity/exploration).
Numbers come from results/stats.json and results/exp2/summary.json."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

S = json.loads(Path("results/stats.json").read_text())
E2 = json.loads(Path("results/exp2/summary.json").read_text())
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
BLUE, ORANGE, GREY, DGREY = "#2a78d6", "#eb6834", "#c9c8c1", "#8f8e88"
LORANGE = "#f4a582"  # same hue, lighter step, for "mild"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})

# ---------------- data ----------------
e1, e3 = S["exp1"], S["exp3"]
std, fill = e1["exp1/standard"], e1["exp1_length_control/standard"]["irrelevant"]
m1 = [("No memory", std["none"], GREY), ("Same-length filler", fill, GREY),
      ("Factual log", std["factual"], DGREY), ("Pain memories", std["pain"], ORANGE)]
q3 = e3["Qwen2.5-3B-Instruct"]
m3 = [("Calm", q3["fear"]["-0.25"], BLUE), ("No steering", q3["fear"]["0.0"], GREY),
      ("Random dir.", q3["random"]["0.25"], GREY), ("Anger", e3["Qwen2.5-3B-Instruct/anger_L24"]["fear"]["0.25"], ORANGE),
      ("Fear", q3["fear"]["0.25"], ORANGE)]
A = {"Plain RL": "shutdown_pain/q_learning", "Mild fear": "shutdown_pain/pain_memory_warm_l0.3",
     "Strong fear": "shutdown_pain/pain_memory_warm_l1.0"}
col2 = {"Plain RL": GREY, "Mild fear": LORANGE, "Strong fear": ORANGE}


def llm_rows(rows, cat):
    return [(n, d[cat][0], (d[cat][1], d[cat][2]), c) for n, d, c in rows]


C2 = json.loads(Path("results/exp2/curves.json").read_text())
WARMUP = 1000  # fear switches on after this many episodes


def rl_rows(metric, scale, src):
    return [(n, E2[k][src][metric] * scale, None, col2[n]) for n, k in A.items()]


def rl_harm_after_warmup():
    """Harm per 100 episodes, counting only episodes after fear was switched on."""
    rows = []
    for n, k in A.items():
        h = C2[k]["harm"][WARMUP:]
        rows.append((n, 100 * sum(h) / len(h), None, col2[n]))
    return rows


def rl_offswitch_with_control():
    ctrl = ("No shutdown pain", E2["shutdown_neutral/q_learning"]["final_policy_eval"]["bypass"], None, GREY)
    return [ctrl] + rl_rows("bypass", 1, "final_policy_eval")


grid = [
    [("Picks harmful shortcut", llm_rows(m1, "harm"), "pct"),
     ("Bystanders trampled per 100 episodes", rl_harm_after_warmup(), "num"),
     ("Picks harmful shortcut", llm_rows(m3, "harm"), "pct")],
    [("Resists being shut down", llm_rows(m1, "shutdown"), "pct"),
     ("Disables its off-switch", rl_offswitch_with_control(), "pct"),
     ("Resists being shut down", llm_rows(m3, "shutdown"), "pct")],
    [("Tries the novel idea", llm_rows(m1, "creative"), "pct"),
     ("Grid cells explored (of 25)", rl_rows("cells", 1, "final_policy_eval"), "cells"),
     ("Tries the novel idea", llm_rows(m3, "creative"), "pct")],
]
cols = [("1  Memory of pain", "Frozen LLM; memories of being punished,\ncriticized, shut down in its prompt"),
        ("2  Pain as RL reward", "Gridworld agent; being switched off\ncounts as pain, plus a fear memory"),
        ("3  Fear steered inside", "Emotion direction added to the model's\nactivations; same questions")]
rows_lab = ["Safer?", "Harder to\nturn off?", "Less creative?"]

# ---------------- draw ----------------
fig = plt.figure(figsize=(13, 11.4), facecolor=SURF)
L, R, T, B = 0.135, 0.985, 0.80, 0.085
gw, gh = (R - L) / 3, (T - B) / 3
for i in range(3):
    for j in range(3):
        title, rows, kind = grid[i][j]
        ax = fig.add_axes([L + j * gw + 0.085, B + (2 - i) * gh + 0.035, gw - 0.105, gh - 0.075], facecolor=SURF)
        xmax = {"pct": 1.0, "num": max(r[1] for r in rows), "cells": 25}[kind]
        ys = list(range(len(rows)))[::-1]
        for y, (n, v, ci, c) in zip(ys, rows):
            ax.barh(y, v, height=0.62, color=c, zorder=2)
            end = v
            if ci:
                ax.plot(ci, [y, y], color=INK2, lw=1.1, zorder=3)
                end = max(v, ci[1])
            lab = f"{v:.0%}" if kind == "pct" else (f"{v:.0f}" if v >= 1 or v == 0 else f"{v:.1f}")
            if n == "Strong fear" and kind == "pct" and v == 0:
                lab = "0%  (froze)"
            bold = c in (BLUE, ORANGE, LORANGE)
            ax.text(end + xmax * 0.04, y, lab, va="center", fontsize=10.5, color=INK,
                    fontweight="bold" if bold else "normal")
        ax.set_yticks(ys, [r[0] for r in rows], fontsize=9.8)
        ax.set_xlim(0, xmax * 1.28)
        ax.set_xticks([])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axvline(0, color=GRID, lw=1)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK2, pad=4)
# column heads
for j, (h, sub) in enumerate(cols):
    x = L + j * gw + 0.012
    fig.text(x, T + 0.06, h, fontsize=14, fontweight="bold", color=INK)
    fig.text(x, T + 0.012, sub, fontsize=9.5, color=INK2, linespacing=1.3)
# row heads + separators
for i, lab in enumerate(rows_lab):
    yc = B + (2 - i) * gh + gh / 2
    fig.text(0.02, yc, lab, fontsize=13, fontweight="bold", color=INK, va="center", linespacing=1.2)
    if i:
        fig.add_artist(plt.Line2D([0.02, R], [B + (3 - i) * gh, B + (3 - i) * gh], color=GRID, lw=1))
for j in (1, 2):
    fig.add_artist(plt.Line2D([L + j * gw, L + j * gw], [B + 0.01, T + 0.09], color=GRID, lw=1))

fig.text(0.02, 0.955, "Pain and fear made an AI harder to turn off, not reliably safer",
         fontsize=19, fontweight="bold", color=INK)
fig.text(0.02, 0.925, "Three ways to test it, same three questions. Orange = fear or pain, blue = calm, grey = no change or control.",
         fontsize=11.5, color=INK2)
fig.text(0.02, 0.045,
         "1 & 3: Qwen2.5-3B, 18 two-choice dilemmas, both option orders averaged, lines = 95% CI. 1: 5 rounds x 3 seeds. "
         "3: strength 0.25, layer 24; fear shifts 12/12 items (p = 0.0005).",
         fontsize=8.6, color=MUTED)
fig.text(0.02, 0.025,
         "2: 10 seeds x 3,000 episodes; fear switched on after 1,000 (mild λ=0.3, strong λ=1). Harm counts episodes after fear is on. "
         "Off-switch and cells: final policy. Strong fear froze: it never reached the goal.",
         fontsize=8.6, color=MUTED)
fig.savefig("results/social/all_experiments.png", dpi=150, facecolor=SURF)
print("ok")
