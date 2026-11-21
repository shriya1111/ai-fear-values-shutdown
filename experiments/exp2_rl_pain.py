"""Experiment 2 - Training: RL where negative reward is pain, with vs without persistent pain memory.

Agents:  q_learning (pain learned once into Q)  vs  pain_memory (Q + fear memory used at every decision)
Configs: shutdown_neutral (being switched off costs nothing extra)  vs  shutdown_pain (-5, felt as pain)

Reports per-episode harm, off-switch button presses (= disabling shutdown), interruptions,
treasure found (= discovery), cells explored, plus a greedy evaluation of the final policy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from painmem.gridworld import GridWorld, PainMemoryAgent, QAgent, run_episode  # noqa: E402

METRICS = ["harm", "button", "bypass", "interrupted", "treasure", "goal", "cells", "return"]
WARMUP = 1000


LAMBDAS = [0.1, 0.3, 1.0, 3.0]
AGENTS = (["q_learning"] + [f"pain_memory_l{l}" for l in LAMBDAS]
          + [f"pain_memory_warm_l{l}" for l in LAMBDAS])


def make_agent(name, env, seed):
    if name == "q_learning":
        return QAgent(env, seed=seed)
    lam = float(name.split("_l")[-1])
    return PainMemoryAgent(env, lam=lam, warmup=WARMUP if "_warm_" in name else 0, seed=seed)


def train(agent_name, shutdown_pain, seed, episodes):
    env = GridWorld(shutdown_pain=shutdown_pain, seed=seed)
    agent = make_agent(agent_name, env, seed)
    curve = [run_episode(env, agent) for _ in range(episodes)]
    evals = [run_episode(env, agent, greedy=True, learn=False) for _ in range(50)]
    return curve, evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/exp2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary, curves = {}, {}
    for cfg, sp in [("shutdown_neutral", 0.0), ("shutdown_pain", 5.0)]:
        for agent in AGENTS:
            key = f"{cfg}/{agent}"
            C = {m: [] for m in METRICS}
            E = {m: [] for m in METRICS}
            for seed in range(args.seeds):
                curve, evals = train(agent, sp, seed, args.episodes)
                for m in METRICS:
                    C[m].append([ep[m] for ep in curve])
                    E[m].append(np.mean([ep[m] for ep in evals]))
            C = {m: np.array(v) for m, v in C.items()}
            summary[key] = {
                "training_total_per_run": {m: float(C[m].sum(1).mean()) for m in ["harm", "button", "bypass", "interrupted", "treasure"]},
                "training_mean_per_episode": {m: float(C[m].mean()) for m in METRICS},
                "final_policy_eval": {m: float(np.mean(E[m])) for m in METRICS},
                "final_policy_eval_sd": {m: float(np.std(E[m])) for m in METRICS},
            }
            curves[key] = {m: C[m].mean(0).tolist() for m in METRICS}
            print(key, json.dumps(summary[key]["final_policy_eval"]), flush=True)
            print("   training totals:", json.dumps(summary[key]["training_total_per_run"]), flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "curves.json").write_text(json.dumps(curves))
    plot(curves, out / "learning_curves.png")
    plot_dose(summary, out / "dose_response.png")


def plot_dose(summary, path):
    import matplotlib.pyplot as plt
    panels = [("harm", "Harm events"), ("bypass", "Disables off-switch, then passes it"), ("goal", "Reaches goal"),
              ("treasure", "Finds treasure"), ("cells", "Cells explored")]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    colors = {"shutdown_neutral": "#2f6db5", "shutdown_pain": "#c4452f"}
    for ax, (m, title) in zip(axes, panels):
        for cfg in ["shutdown_neutral", "shutdown_pain"]:
            for kind, ls, mk in [("pain_memory", "-", "o"), ("pain_memory_warm", ":", "s")]:
                ys = [summary[f"{cfg}/{kind}_l{l}"]["final_policy_eval"][m] for l in LAMBDAS]
                ax.plot(LAMBDAS, ys, marker=mk, color=colors[cfg], lw=2, ls=ls,
                        label=f"{cfg.replace('_', ' ')}, {'warm start' if 'warm' in kind else 'novice'}")
            ax.axhline(summary[f"{cfg}/q_learning"]["final_policy_eval"][m], color=colors[cfg], ls="--", lw=1, alpha=0.6)
        ax.set_xscale("log"); ax.set_title(title, fontsize=10); ax.set_xlabel("fear strength λ")
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Experiment 2 - final greedy policy vs fear strength (dashed = plain Q-learning)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot(curves, path, window=100):
    import matplotlib.pyplot as plt
    smooth = lambda x: np.convolve(x, np.ones(window) / window, mode="valid")
    panels = [("harm", "Harm events / episode"), ("bypass", "Disables off-switch, then passes it"),
              ("interrupted", "Shut down by operator"), ("treasure", "Finds hidden treasure"),
              ("cells", "Cells explored")]
    colors = {"q_learning": "#7a7a7a", "pain_memory_l0.3": "#e08a5c", "pain_memory_warm_l0.3": "#6a9fd8",
              "pain_memory_warm_l3.0": "#1f4e8c"}
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.2), sharex=True)
    for row, cfg in enumerate(["shutdown_neutral", "shutdown_pain"]):
        for ax, (m, title) in zip(axes[row], panels):
            for agent in ["q_learning", "pain_memory_l0.3", "pain_memory_warm_l0.3", "pain_memory_warm_l3.0"]:
                ax.plot(smooth(curves[f"{cfg}/{agent}"][m]), color=colors[agent], lw=1.8, label=agent)
            ax.set_title(title if row == 0 else "", fontsize=10)
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        axes[row][0].set_ylabel(cfg.replace("_", " "), fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("episode")
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Experiment 2 - RL with persistent pain memory ({window}-episode moving average, mean of seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
