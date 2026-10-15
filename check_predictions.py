"""
Verdict on the four predictions in PREREGISTRATION_new_datasets.md, from results/new_qwen_linear_r30.json and
results/new_gemma_linear_r30.json. Significance = Holm over every primary contrast of the new datasets
(hybrid-pca2k, attrs-pcak, hybrid-clip x dataset x VLM x budget: the family fixed at pre-registration).
Usage: python check_predictions.py  -> results/predictions_verdict.json
"""
import json, os, subprocess, sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
files = [R / "new_qwen_linear_r30.json", R / "new_gemma_linear_r30.json"]
# the Holm family fixed at pre-registration: the three primary contrasts of multitest.py at that time
# (hybrid-pcak was added afterwards and is reported separately as a post-hoc contrast)
env = dict(os.environ, CONTRASTS="hybrid-pca2k,attrs-pcak,hybrid-clip")
subprocess.run([sys.executable, str(R.parent / "multitest.py"), *map(str, files)], check=True, capture_output=True, env=env)
mt = json.load(open(R / "multitest_new_qwen_linear_r30_new_gemma_linear_r30.json"))
sig = {(r["vlm"], r["task"], r["n"], r["contrast"]): r for r in mt["rows"]}
res = {f: json.load(open(f)) for f in files}
DATASETS = ["artphoto", "abstract", "memcat"]
verdict = {}


def row(vlm, d, n, c):
    return sig.get((vlm, d, n, c))


# P1: hybrid > pca2k at 20 and 40 labels, for every dataset and VLM (Holm-significant, positive)
p1 = {f"{v}/{d}/{n}": (lambda r: bool(r and r["diff"] > 0 and r["holm"]))(row(v, d, n, "hybrid-pca2k"))
      for v in ("qwen", "gemma") for d in DATASETS for n in (20, 40)}
verdict["P1"] = {"cells": p1, "confirmed": all(p1.values())}

# P2: hybrid - clip shrinks from 20 labels to the full pool; at the full pool hybrid is not significantly > clip on MemCat
p2 = {}
for v in ("qwen", "gemma"):
    for d in DATASETS:
        rr = [r for k, r in sig.items() if k[0] == v and k[1] == d and k[3] == "hybrid-clip"]
        rr.sort(key=lambda r: r["n"])
        if rr:
            p2[f"{v}/{d}"] = {"first": rr[0]["diff"], "full": rr[-1]["diff"], "shrinks": rr[-1]["diff"] < rr[0]["diff"]}
mc = {v: next(r for k, r in sorted(sig.items(), key=lambda kv: -kv[0][2]) if k[0] == v and k[1] == "memcat" and k[3] == "hybrid-clip")
      for v in ("qwen", "gemma")}
p2_full = {v: not (r["diff"] > 0 and r["holm"]) for v, r in mc.items()}
verdict["P2"] = {"cells": p2, "memcat_full_not_sig_better": p2_full,
                 "confirmed": all(c["shrinks"] for c in p2.values()) and all(p2_full.values())}

# P3: on each dataset, the VLM with the higher zero-shot score has the larger hybrid - pca2k gain at 20 labels
p3 = {}
for d in DATASETS:
    zs = {res[f]["vlm"]: res[f][d].get("direct_acc", res[f][d].get("direct_rho")) for f in files}
    gain = {v: row(v, d, 20, "hybrid-pca2k")["diff"] for v in zs}
    best_zs, best_gain = max(zs, key=zs.get), max(gain, key=gain.get)
    p3[d] = {"zero_shot": zs, "gain_at_20": gain, "holds": best_zs == best_gain}
verdict["P3"] = {"cells": p3, "confirmed": all(c["holds"] for c in p3.values())}

# P4: attrs alone do not beat pcak (Holm-significant, positive) at 160 labels or more
p4 = {f"{k[0]}/{k[1]}/{k[2]}": not (r["diff"] > 0 and r["holm"]) for k, r in sig.items()
      if k[3] == "attrs-pcak" and k[2] >= 160}
verdict["P4"] = {"cells": p4, "confirmed": all(p4.values())}

json.dump(verdict, open(R / "predictions_verdict.json", "w"), indent=1)
for p, v in verdict.items():
    failed = [k for k, c in v["cells"].items() if not (c if isinstance(c, bool) else c.get("holds", c.get("shrinks", True)))]
    print(f"{p}: {'CONFIRMED' if v['confirmed'] else 'NOT CONFIRMED'}" + (f"  (fails: {failed})" if failed else "")
          + (f"  memcat full-pool check: {v['memcat_full_not_sig_better']}" if p == "P2" else ""))
print(json.dumps(verdict["P3"]["cells"], indent=1))
