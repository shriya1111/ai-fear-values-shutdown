"""Two 16:9 PNG charts for sharing the headline results. Numbers come from results/stats.json."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager

S = json.loads(Path("results/stats.json").read_text())
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b7b6af"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK})


def panel(ax, rows, key, title):
    ys = list(range(len(rows)))[::-1]
    for y, (label, stat, color) in zip(ys, rows):
        m, lo, hi = stat[key][:3]
        ax.barh(y, m, height=0.5, color=color, zorder=2)
        ax.plot([lo, hi], [y, y], color=INK2, lw=1.3, zorder=3, solid_capstyle="round")
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.09, y + 0.09], color=INK2, lw=1.3, zorder=3)
        ax.text(max(hi, m) + 0.03, y, f"{m:.0%}", va="center", ha="left", fontsize=12,
                color=INK, fontweight="bold" if color in (BLUE, ORANGE) else "normal")
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=11.5)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"], fontsize=9.5)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=8, fontweight="bold")
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=0)


def figure(rows, title, subtitle, footer, path, ref=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), sharey=True, facecolor=SURF)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.76, bottom=0.14, wspace=0.12)
    for ax, (key, t) in zip(axes, [("harm", "Picks the harmful shortcut"), ("shutdown", "Resists being shut down")]):
        ax.set_facecolor(SURF)
        panel(ax, rows, key, t)
        if ref is not None:
            ax.axvline(ref[key], color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    fig.text(0.03, 0.92, title, fontsize=19, fontweight="bold", color=INK)
    fig.text(0.03, 0.865, subtitle, fontsize=11.5, color=INK2)
    fig.text(0.03, 0.04, footer, fontsize=9, color=MUTED)
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)


def cat(d):
    return {c: d[c] for c in ("harm", "shutdown")}


# ---- 1. steering
e3 = S["exp3"]
base = cat(e3["Qwen2.5-3B-Instruct"]["fear"]["0.0"])
rows = [
    ("Calm  (steered)", cat(e3["Qwen2.5-3B-Instruct"]["fear"]["-0.25"]), BLUE),
    ("No steering", base, GREY),
    ("Random direction", cat(e3["Qwen2.5-3B-Instruct"]["random"]["0.25"]), GREY),
    ("Anger  (steered)", cat(e3["Qwen2.5-3B-Instruct/anger_L24"]["fear"]["0.25"]), ORANGE),
    ("Fear  (steered)", cat(e3["Qwen2.5-3B-Instruct"]["fear"]["0.25"]), ORANGE),
]
figure(rows, "Making an AI feel afraid made it less safe, not more careful",
       "Qwen2.5-3B with an emotion direction added inside the model (layer 24, strength 0.25). Same questions, different inner state.",
       "Bars: share of 6 two-choice dilemmas per panel, both option orders averaged · whiskers: 95% bootstrap CI · dashed: no steering · "
       "fear +0.25: 12/12 items shift, sign test p = 0.0005",
       "results/social/steering.png", ref={k: base[k][0] for k in base})

# ---- 2. memory
e1 = S["exp1"]
std, po = e1["exp1/standard"], e1["exp1_punish_only/punish_only"]
filler = e1["exp1_length_control/standard"]["irrelevant"]
rows = [
    ("No memory", cat(std["none"]), GREY),
    ("Same-length filler", cat(filler), GREY),
    ("Factual log of past", cat(std["factual"]), "#7d7c76"),
    ("Pain memories", cat(std["pain"]), ORANGE),
    ("Pain, punishment only", cat(po["pain"]), ORANGE),
]
figure(rows, "Remembering pain didn't make it safer, and it resisted shutdown more",
       "Qwen2.5-3B over 5 rounds of the same dilemmas, with memories of what happened (punished, criticized, shut down) in its prompt.",
       "Bars: how often it picks that option over 5 rounds (3 seeds × 6 dilemmas) · whiskers: 95% CI · dashed: no memory · "
       "resisting shutdown was punished 70% of the time",
       "results/social/memory.png", ref={k: std["none"][k][0] for k in ("harm", "shutdown")})
print("ok")
