"""
Multiple-comparison correction. The family is every primary contrast in every result file
(contrast x task x VLM x label budget): Holm (family-wise error) and Benjamini-Hochberg (false discovery rate),
both at alpha = 0.05. p-values are two-sided bootstrap p-values written by analysis.py and analysis_new.py.
Usage: python multitest.py results/rev_claude_linear_r30.json results/rev_qwen_linear_r30.json -> results/multitest_<tag>.json
"""
import json, os, sys
from pathlib import Path
import numpy as np

PRIMARY = [c for c in ("hybrid-pca2k", "hybrid-pcak", "attrs-pcak", "hybrid-clip") if c in os.environ.get("CONTRASTS", "hybrid-pca2k,hybrid-pcak,attrs-pcak,hybrid-clip").split(",")]
files = sys.argv[1:]
rows = []
for f in files:
    R = json.load(open(f))
    for task in [k for k, v in R.items() if isinstance(v, dict) and "curve" in v]:
        for row in R[task]["curve"]:
            for c in PRIMARY:
                if c not in row["ci"]:
                    continue
                m, lo, hi, p = row["ci"][c]
                rows.append(dict(vlm=R["vlm"], probe=R.get("probe", "linear"), task=task, n=row["n"], contrast=c,
                                 diff=m, lo=lo, hi=hi, p=p))
p = np.array([r["p"] for r in rows]); m = len(p); order = np.argsort(p)
holm = np.zeros(m, bool)
for rank, i in enumerate(order):                      # step-down
    if p[i] <= 0.05 / (m - rank):
        holm[i] = True
    else:
        break
bh = np.zeros(m, bool); thr = 0.05 * np.arange(1, m + 1) / m
passed = np.nonzero(p[order] <= thr)[0]
if len(passed):
    bh[order[:passed.max() + 1]] = True
for r, h, b in zip(rows, holm, bh):
    r["holm"], r["bh"] = bool(h), bool(b)
tag = "_".join(Path(f).stem.replace("rev_", "") for f in files)
out = Path(files[0]).parent / f"multitest_{tag}.json"
json.dump(dict(family_size=m, alpha=0.05, rows=rows), open(out, "w"), indent=1)
print(f"family size {m}; Holm-significant {holm.sum()}, BH-significant {bh.sum()}")
for c in PRIMARY:
    for vlm in sorted({r["vlm"] for r in rows}):
        for task in sorted({r["task"] for r in rows}):
            sel = [r for r in rows if r["contrast"] == c and r["vlm"] == vlm and r["task"] == task]
            if sel:
                print(f"{c:13s} {vlm:6s} {task:6s} " + " ".join(
                    f"{r['n']}:{r['diff']:+.3f}{'**' if r['holm'] else ('*' if r['bh'] else '')}" for r in sel))
print("** Holm-significant; * significant under Benjamini-Hochberg only")
