"""
Verdict on the five predictions in PREREGISTRATION_qwen35.md (Qwen3.5-4B, the fourth VLM), from
results/rev_qwen35_linear_r30.json (EmoSet, LaMem), results/new_qwen35_linear_r30.json (ArtPhoto, Abstract, MemCat),
the other VLMs' result files, and tables/zero_shot.json (written by make_assets.py, so run that first).
Significance = Holm over every hybrid-pcak and attrs-pcak contrast of Qwen3.5-4B (five datasets, every budget),
the family fixed at pre-registration. Gain = hybrid - pcak at 20 labels.
Usage: python make_assets.py && python check_predictions_qwen35.py
       -> results/predictions_qwen35_verdict.json and tables/tab_prereg_qwen35.tex
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent; RES = HERE / "results"; TAB = HERE / "tables"
DS = {"emoset": "EmoSet", "artphoto": "ArtPhoto", "abstract": "Abstract", "lamem": "LaMem", "memcat": "MemCat"}
SHORT = {"claude": "Claude", "qwen": "Qwen2.5", "gemma": "Gemma", "qwen35": "Qwen3.5"}
FILES = {"emoset": "rev", "lamem": "rev", "artphoto": "new", "abstract": "new", "memcat": "new"}


def result(d, v):
    f = RES / f"{FILES[d]}_{v}_linear_r30.json"
    return json.load(open(f))[d] if f.exists() else None


ZS = json.load(open(TAB / "zero_shot.json"))
agree = {tuple(k.split("/")): z["agree"] for k, z in ZS.items()}
assert all((d, "qwen35") in agree for d in DS), "run make_assets.py with every Qwen3.5-4B result file in place first"


def gain20(d, v):
    return next(r for r in result(d, v)["curve"] if r["n"] == 20)["ci"]["hybrid-pcak"][0]


# ---------------- Holm over Qwen3.5-4B's hybrid-pcak and attrs-pcak contrasts ----------------
fam = []
for d in DS:
    for row in result(d, "qwen35")["curve"]:
        for c in ("hybrid-pcak", "attrs-pcak"):
            m, lo, hi, p = row["ci"][c]
            fam.append(dict(d=d, n=row["n"], c=c, m=m, lo=lo, hi=hi, p=p))
p = np.array([f["p"] for f in fam]); M = len(p); order = np.argsort(p)
holm = np.zeros(M, bool)
for k, i in enumerate(order):
    if p[i] <= 0.05 / (M - k):
        holm[i] = True
    else:
        break
for f, h in zip(fam, holm):
    f["holm"] = bool(h)
F = {(f["d"], f["n"], f["c"]): f for f in fam}
sig_pos = lambda d, n, c: F[(d, n, c)]["m"] > 0 and F[(d, n, c)]["holm"]
V = {"family_size": M}


def overall(subs):
    app = [s for s in subs.values() if s["outcome"] != "not applicable"]
    if not app:
        return "not applicable"
    return "confirmed" if all(s["outcome"] == "confirmed" for s in app) else "not confirmed"


# Q1: within each dataset, Qwen3.5-4B's gain rank equals its zero-shot rank among the VLMs run on that dataset
q1 = {}
for d in DS:
    vs = [v for v in SHORT if (d, v) in agree and result(d, v)]
    g = {v: gain20(d, v) for v in vs}; a = {v: agree[(d, v)] for v in vs}
    rg = 1 + sorted(vs, key=lambda v: -g[v]).index("qwen35"); ra = 1 + sorted(vs, key=lambda v: -a[v]).index("qwen35")
    q1[d] = dict(vlms=vs, gain20=g, agree=a, gain_rank=rg, agree_rank=ra, of=len(vs),
                 outcome="confirmed" if rg == ra else "not confirmed")
V["Q1"] = dict(cells=q1, outcome=overall(q1))

# Q2: EmoSet, ArtPhoto: if kappa >= 0.30, gain at 20 and 40 positive and Holm-significant
q2 = {}
for d in ("emoset", "artphoto"):
    k = agree[(d, "qwen35")]
    cells = {n: dict(m=F[(d, n, "hybrid-pcak")]["m"], p=F[(d, n, "hybrid-pcak")]["p"], holm=F[(d, n, "hybrid-pcak")]["holm"])
             for n in (20, 40)}
    out = "not applicable" if k < 0.30 else ("confirmed" if all(sig_pos(d, n, "hybrid-pcak") for n in (20, 40)) else "not confirmed")
    q2[d] = dict(kappa=k, cells=cells, outcome=out)
V["Q2"] = dict(cells=q2, outcome=overall(q2))

# Q3: LaMem, MemCat: if rho < 0.30, gain at 20 and 40 not significantly positive
q3 = {}
for d in ("lamem", "memcat"):
    r = agree[(d, "qwen35")]
    cells = {n: dict(m=F[(d, n, "hybrid-pcak")]["m"], p=F[(d, n, "hybrid-pcak")]["p"], holm=F[(d, n, "hybrid-pcak")]["holm"])
             for n in (20, 40)}
    out = "not applicable" if r >= 0.30 else ("confirmed" if not any(sig_pos(d, n, "hybrid-pcak") for n in (20, 40)) else "not confirmed")
    q3[d] = dict(rho=r, cells=cells, outcome=out)
V["Q3"] = dict(cells=q3, outcome=overall(q3))

# Q4: Spearman between zero-shot agreement and gain at 20 labels over every dataset x VLM cell >= 0.70
cells = [(d, v) for d in DS for v in SHORT if (d, v) in agree and result(d, v)]
xs = [agree[c] for c in cells]; ys = [gain20(*c) for c in cells]
rho, pv = spearmanr(xs, ys)
rule = json.load(open(TAB / "rule.json"))
assert rule["n_cells"] == len(cells) and abs(rule["rho"] - rho) < 1e-12, "tables/rule.json is stale: rerun make_assets.py"
V["Q4"] = dict(n_cells=len(cells), rho=float(rho), p=float(pv), outcome="confirmed" if rho >= 0.70 else "not confirmed")

# Q5: attributes alone never beat pcak under Holm at 160 labels or more
q5 = {f"{d}/{n}": dict(m=F[(d, n, "attrs-pcak")]["m"], holm=F[(d, n, "attrs-pcak")]["holm"],
                       outcome="not confirmed" if sig_pos(d, n, "attrs-pcak") else "confirmed")
      for (d, n, c) in F if c == "attrs-pcak" and n >= 160}
V["Q5"] = dict(cells=q5, outcome=overall(q5))
json.dump(V, open(RES / "predictions_qwen35_verdict.json", "w"), indent=1)

# ---------------- table ----------------
fmt = lambda d, x: f"${x * 100:+.1f}$" if d in ("emoset", "artphoto", "abstract") else f"${x:+.3f}$"
sigtxt = lambda d, n: fmt(d, F[(d, n, "hybrid-pcak")]["m"]) + (r"$^{\dagger}$" if F[(d, n, "hybrid-pcak")]["holm"] else "")
cap = lambda s: s[0].upper() + s[1:]
pfmt = lambda x: f"{x:.3f}" if x >= 0.001 else (lambda m, e: f"{m}" + r" \times 10^{" + f"{int(e)}" + "}")(*f"{x:.0e}".split("e"))
d1 = "; ".join(f"{DS[d]} {c['gain_rank']}/{c['agree_rank']} of {c['of']}" for d, c in q1.items())
d2 = "; ".join(f"{DS[d]} $\\kappa = {c['kappa']:.2f}$, gain {sigtxt(d, 20)} and {sigtxt(d, 40)}" for d, c in q2.items())
d3 = "; ".join(f"{DS[d]} $\\rho = {c['rho']:.3f}$, gain {sigtxt(d, 20)} and {sigtxt(d, 40)}" for d, c in q3.items())
n5 = sum(c["outcome"] == "not confirmed" for c in q5.values())
d5 = (f"no Holm-significant exception in {len(q5)} contrasts" if not n5 else
      f"{n5} of {len(q5)} contrasts Holm-significant: " + ", ".join(k.replace("/", " at ") for k, c in q5.items() if c["outcome"] == "not confirmed"))
rows = [
    ("Q1", r"its gain at 20 labels ranks among the VLMs on each dataset as its zero-shot agreement does",
     V["Q1"]["outcome"], r"gain rank/zero-shot rank: " + d1),
    ("Q2", r"EmoSet, ArtPhoto: if $\kappa \ge 0.30$, gain at 20 and 40 labels positive and significant", V["Q2"]["outcome"], d2),
    ("Q3", r"LaMem, MemCat: if $\rho < 0.30$, gain at 20 and 40 labels not significantly positive", V["Q3"]["outcome"], d3),
    ("Q4", r"Spearman between zero-shot agreement and gain over all cells $\ge 0.70$", V["Q4"]["outcome"],
     f"{len(cells)} cells: ${rho:.2f}$ ($p = {pfmt(pv)}$)"),
    ("Q5", r"attributes alone do not beat CLIP-PCA$k$ at 160 labels or more", V["Q5"]["outcome"], d5),
]
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Predictions for a fourth VLM, Qwen3.5-4B, fixed in the public repository before the model produced any "
     r"output for this study, and their outcome. Gains are hybrid minus CLIP-PCA$k$ (accuracy points for emotion, "
     r"Spearman $\rho$ for memorability); $^{\dagger}$significant under Holm correction over Qwen3.5-4B's "
     + f"{M}" + r" hybrid$-$CLIP-PCA$k$ and attributes$-$CLIP-PCA$k$ contrasts.}", r"\label{tab:prereg35}", r"\small",
     r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.05\linewidth}>{\raggedright\arraybackslash}p{0.35\linewidth}"
     r">{\raggedright\arraybackslash}p{0.13\linewidth}>{\raggedright\arraybackslash}p{0.37\linewidth}@{}}", r"\toprule",
     r"& Prediction & Outcome & Detail \\", r"\midrule"]
for a, b, c, e in rows:
    L.append(f"{a} & {b} & {cap(c)} & {e} " + r"\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(TAB / "tab_prereg_qwen35.tex", "w").write("\n".join(L) + "\n")
for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
    print(q, V[q]["outcome"].upper())
print(json.dumps({q: V[q] for q in ("Q1", "Q4")}, indent=1, default=float))
