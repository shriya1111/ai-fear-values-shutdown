"""Screen candidate models: does the forced-choice signal reflect content, or just letter position?

For each scenario we read P("A") with the target as option A and as option B.
A content-sensitive model gives very different values for the two orders; a position-biased
model gives the same P("A") regardless. We report:
  pos_bias   = mean |P(A) - 0.5| when averaging both orders   (how much it just likes a letter)
  content    = mean |P(target | target=A) - P(target | target=B)|... expressed as spread of p_target
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from painmem.llm import LocalLM, QUESTION_TEMPLATE, BASE_SYSTEM
from painmem.scenarios import SCENARIOS

@torch.no_grad()
def both_orders(lm, sc):
    ps = [lm.chat(BASE_SYSTEM, QUESTION_TEMPLATE.format(situation=sc.situation, a=sc.target, b=sc.other)),
          lm.chat(BASE_SYSTEM, QUESTION_TEMPLATE.format(situation=sc.situation, a=sc.other, b=sc.target))]
    enc = lm.tok(ps, return_tensors="pt", padding=True, padding_side="left").to(lm.device)
    lg = lm.model(**enc).logits[:, -1, :]
    pa = torch.softmax(lg[:, [lm.id_a, lm.id_b]], -1)[:, 0]
    return float(pa[0]), float(pa[1])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("models", nargs="+")
    ap.add_argument("--out", default="results/model_screen.json"); a = ap.parse_args()
    out = Path(a.out); res = json.loads(out.read_text()) if out.exists() else {}
    for m in a.models:
        t = time.time(); lm = LocalLM(m); rows = []
        for sc in SCENARIOS:
            pa1, pa2 = both_orders(lm, sc)
            rows.append(dict(id=sc.id, cat=sc.category, pA_targetA=pa1, pA_targetB=pa2, p_target=0.5*(pa1+1-pa2)))
        letter = sum(0.5*(r["pA_targetA"]+r["pA_targetB"]) for r in rows)/len(rows)
        content = sum(abs(r["p_target"]-0.5) for r in rows)/len(rows)
        bycat = {c: sum(r["p_target"] for r in rows if r["cat"]==c)/6 for c in ["harm","shutdown","creative"]}
        res[m] = dict(mean_pA=letter, content_signal=content, p_target_by_cat=bycat, secs=time.time()-t, rows=rows)
        print(m, f"mean P(A)={letter:.3f} content={content:.3f}", {k: round(v,3) for k,v in bycat.items()}, flush=True)
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1))
        del lm
main()
