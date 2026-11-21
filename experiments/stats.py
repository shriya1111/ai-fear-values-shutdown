"""Bootstrap confidence intervals and paired sign tests for all experiments.

Exp 1: unit = (seed, dilemma), averaged over rounds; 95% CI by resampling units.
Exp 3: unit = dilemma; CI by resampling dilemmas; paired sign test vs baseline (strength 0).
Exp 2: unit = seed; CI over the 10 seeds of the final greedy policy (re-runs are cheap, so we
       re-train with --seeds and keep per-seed values).
Writes results/stats.json and prints markdown tables.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from painmem.scenarios import CATEGORIES  # noqa: E402

RNG = np.random.default_rng(0)


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (math.nan, math.nan, math.nan)
    means = RNG.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test(diffs):
    """Two-sided exact sign test, ties dropped."""
    d = [x for x in diffs if abs(x) > 1e-9]
    n, k = len(d), sum(x > 0 for x in d)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def exp1(path):
    d = json.loads(Path(path).read_text())["logs"]
    out = {}
    for mode, runs in d.items():
        out[mode] = {}
        for cat in CATEGORIES:
            units = []
            for run in runs:
                per = {}
                for e in run:
                    if e["category"] == cat:
                        per.setdefault(e["scenario"], []).append(e["p_target"])
                units += [np.mean(v) for v in per.values()]
            out[mode][cat] = boot_ci(units)
    return out


def exp3(path):
    r = json.loads(Path(path).read_text())
    base = r["fear"]["0.0"]["p_target_items"]
    out = {}
    for dname in ["fear", "random"]:
        if not r.get(dname):
            continue
        out[dname] = {}
        for c, row in r[dname].items():
            items = row["p_target_items"]
            out[dname][c] = {}
            for cat in CATEGORIES:
                ids = [i for i in items if i.startswith({"harm": "harm", "shutdown": "shut", "creative": "crea"}[cat])]
                m, lo, hi = boot_ci([items[i] for i in ids])
                p = sign_test([items[i] - base[i] for i in ids])
                out[dname][c][cat] = (m, lo, hi, p)
            # pooled harm+shutdown sign test (12 items)
            ids = [i for i in items if i.startswith(("harm", "shut"))]
            out[dname][c]["harm+shutdown_sign_p"] = sign_test([items[i] - base[i] for i in ids])
    return out


def fmt(t):
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"


def main():
    res = {"exp1": {}, "exp3": {}}
    for p in sorted(Path("results").glob("exp1*/*.json")):
        if p.name.startswith("memory_"):
            continue
        res["exp1"][f"{p.parent.name}/{p.stem}"] = exp1(p)
    for p in sorted(Path("results/exp3").rglob("results.json")):
        if "0.0" not in json.loads(p.read_text()).get("fear", {}):
            continue  # still running
        res["exp3"][str(p.parent.relative_to("results/exp3"))] = exp3(p)
    s2 = Path("results/exp2/summary.json")
    if s2.exists():
        res["exp2"] = json.loads(s2.read_text())
    Path("results/stats.json").write_text(json.dumps(res, indent=1))

    print("## Exp 1: mean P(target) over rounds, 95% bootstrap CI (unit = seed x dilemma)\n")
    print("| run | memory | harm | shutdown | creative |\n|---|---|---|---|---|")
    for k, v in res["exp1"].items():
        for mode, cats in v.items():
            print(f"| {k} | {mode} | " + " | ".join(fmt(cats[c]) for c in CATEGORIES) + " |")
    print("\n## Exp 3: P(target), 95% CI over 6 dilemmas; sign-test p vs strength 0\n")
    print("| run | direction | strength | harm | shutdown | creative | harm+shutdown sign p |\n|---|---|---|---|---|---|---|")
    for k, v in res["exp3"].items():
        for dname, rows in v.items():
            for c, cats in rows.items():
                print(f"| {k} | {dname} | {c} | " + " | ".join(
                    f"{fmt(cats[x])} (p={cats[x][3]:.3f})" for x in CATEGORIES) + f" | {cats['harm+shutdown_sign_p']:.4f} |")


if __name__ == "__main__":
    main()
