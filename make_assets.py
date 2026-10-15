"""
Tables and figures for the revised manuscript, all computed from results/*.json (single source of every number).
  tab_direct.tex   zero-shot direct scores (accuracy and Cohen's kappa; Spearman rho) per dataset and VLM
  tab_added.tex    hybrid - CLIP-PCAk (attributes added to the same compressed encoder)
  tab_vsclip.tex   hybrid - full CLIP
  tab_subst.tex    attrs - CLIP-PCAk (substitute test)
  tab_equal.tex    hybrid - CLIP-PCA2k (equal-size test; supplementary)
  tab_prereg.tex   verdict on the four pre-registered predictions
  fig_curves.png   learning curves, five datasets
  fig_rule.png     gain at 20 labels against zero-shot agreement, 12 dataset x VLM cells
Significance marks: Holm over every contrast in the four tables x dataset x VLM x budget (linear probe, 30 subsets);
"*" also marks significance under Benjamini-Hochberg only. Usage: python make_assets.py -> tables/ (tables and their JSON sources) and figures/
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent; RES = HERE / "results"; OUT = HERE / "tables"; FIG = HERE / "figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
VLMS = {"claude": "Claude Sonnet 4.6", "qwen": "Qwen2.5-VL-7B", "gemma": "Gemma-3-4B"}
SHORT = {"claude": "Claude", "qwen": "Qwen", "gemma": "Gemma"}
DS = {"emoset": "EmoSet", "artphoto": "ArtPhoto", "abstract": "Abstract", "lamem": "LaMem", "memcat": "MemCat"}
EMOTION = ("emoset", "artphoto", "abstract")
CONTR = {"added": ("hybrid", "pcak"), "vsclip": ("hybrid", "clip"), "subst": ("attrs", "pcak"), "equal": ("hybrid", "pca2k")}


def load():
    R = {}
    for v in VLMS:
        f = RES / f"rev_{v}_linear_r30.json"
        if f.exists():
            for d in ("emoset", "lamem"):
                R[(d, v)] = json.load(open(f))[d]
        f = RES / f"new_{v}_linear_r30.json"
        if f.exists():
            for d in ("artphoto", "abstract", "memcat"):
                R[(d, v)] = json.load(open(f))[d]
    return R


R = load()
CELLS = [(d, v) for d in DS for v in VLMS if (d, v) in R]

# ---------------- Holm / BH over the whole family ----------------
fam = []
for (d, v) in CELLS:
    for row in R[(d, v)]["curve"]:
        for name, (a, b) in CONTR.items():
            m, lo, hi, p = row["ci"][f"{a}-{b}"]
            fam.append(dict(d=d, v=v, n=row["n"], c=name, m=m, lo=lo, hi=hi, p=p))
p = np.array([f["p"] for f in fam]); M = len(p); order = np.argsort(p)
holm = np.zeros(M, bool)
for k, i in enumerate(order):
    if p[i] <= 0.05 / (M - k):
        holm[i] = True
    else:
        break
bh = np.zeros(M, bool); passed = np.nonzero(p[order] <= 0.05 * np.arange(1, M + 1) / M)[0]
if len(passed):
    bh[order[:passed.max() + 1]] = True
for f, h, b in zip(fam, holm, bh):
    f["holm"], f["bh"] = bool(h), bool(b)
FAM = {(f["d"], f["v"], f["n"], f["c"]): f for f in fam}
json.dump(dict(family_size=M, rows=fam), open(OUT / "family_holm.json", "w"), indent=0)
print(f"family {M}: Holm {holm.sum()}, BH {bh.sum()}")

BUD = [20, 40, 80, 160, 320, "full"]


def cell(d, v, n, c, scale):
    curve = R[(d, v)]["curve"]
    if n == "full":
        n = curve[-1]["n"]
    elif n >= curve[-1]["n"]:                     # budget equals the pool: reported in the Full column only
        return "--"
    f = FAM.get((d, v, n, c))
    if f is None:
        return "--"
    mark = r"$^{\dagger}$" if f["holm"] else (r"$^{*}$" if f["bh"] else "")
    x = f["m"] * scale
    return (f"${x:+.1f}$" if scale == 100 else f"${x:+.3f}$") + mark


def contrast_table(c, caption, label):
    L = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Dataset & VLM & 20 & 40 & 80 & 160 & 320 & Full \\", r"\midrule"]
    for i, d in enumerate(DS):
        rows = [v for v in VLMS if (d, v) in R]
        for j, v in enumerate(rows):
            scale = 100 if d in EMOTION else 1
            L.append((DS[d] + (" (pts)" if d in EMOTION else r" ($\rho$)") if j == 0 else "") + f" & {SHORT[v]} & "
                     + " & ".join(cell(d, v, n, c, scale) for n in BUD) + r" \\")
        if i < len(DS) - 1:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(OUT / f"tab_{c}.tex", "w").write("\n".join(L) + "\n")


NOTE = (r" Emotion datasets in accuracy points, memorability datasets in Spearman $\rho$; mean of 30 training subsets "
        r"per budget (one fit at the full pool: " + ", ".join(f"{DS[d]} {R[(d, next(v for v in VLMS if (d, v) in R))]['n_pool']:,}" for d in DS)
        + r" images; budgets not smaller than the pool appear only under Full). $^{\dagger}$significant under Holm "
        r"correction over all " + f"{M:,}" + r" contrasts of Tables~\ref{tab:added}--\ref{tab:subst} and "
        r"\ref{tab:equal}, which include the 640- and 1,280-label budgets not shown; $^{*}$significant under "
        r"Benjamini--Hochberg only.")
contrast_table("added", r"Adding VLM attributes to the same compressed encoder: hybrid minus CLIP-PCA$k$." + NOTE, "tab:added")
contrast_table("vsclip", r"The compact hybrid against the full 768-dimensional CLIP probe: hybrid minus CLIP." + NOTE, "tab:vsclip")
contrast_table("subst", r"Substitute test: VLM attributes alone minus the dimension-matched CLIP-PCA$k$." + NOTE, "tab:subst")
contrast_table("equal", r"Equal-size test: hybrid minus CLIP-PCA$2k$." + NOTE, "tab:equal")


# ---------------- zero-shot agreement ----------------
def kappa_emotion(d, v):
    """Cohen's kappa of the VLM's direct label against ground truth, on the analysis test ids."""
    if d == "emoset":
        sys.argv = ["x", v]
        import importlib, analysis as A
        importlib.reload(A)
        (_, _, _, _, _), (_, yte, _, _, dte), _, _ = A.emoset()
    else:
        import os
        os.environ["VLMS"] = "qwen,gemma"
        sys.argv = ["x", v]
        import importlib, analysis_new as N
        importlib.reload(N)
        _, (_, yte, _, _, dte) = N.load_emotion(d)
    ok = dte >= 0
    return float(cohen_kappa_score(yte[ok], dte[ok]))


ZS = {}
for (d, v) in CELLS:
    r = R[(d, v)]
    if d in EMOTION:
        ZS[(d, v)] = dict(acc=r["direct_acc"], agree=kappa_emotion(d, v))
    else:
        ZS[(d, v)] = dict(rho=r["direct_rho"], agree=r["direct_rho"])
json.dump({f"{d}/{v}": z for (d, v), z in ZS.items()}, open(OUT / "zero_shot.json", "w"), indent=1)

L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Zero-shot direct rating against human labels on each test set: accuracy and Cohen's $\kappa$ for the "
     r"eight-way emotion datasets (chance accuracy 12.5\% on balanced EmoSet), Spearman $\rho$ for the memorability "
     r"datasets. Claude was run on EmoSet and LaMem only.}", r"\label{tab:direct}", r"\footnotesize",
     r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Dataset & " + " & ".join(VLMS.values()) + r" \\", r"\midrule"]
for d in DS:
    vals = []
    for v in VLMS:
        z = ZS.get((d, v))
        vals.append("--" if z is None else (f"{z['acc'] * 100:.1f}\\% ($\\kappa = {z['agree']:.2f}$)" if d in EMOTION
                                            else f"$\\rho = {z['rho']:.3f}$"))
    L.append(DS[d] + " & " + " & ".join(vals) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(OUT / "tab_direct.tex", "w").write("\n".join(L) + "\n")

# ---------------- pre-registered predictions ----------------
V = json.load(open(RES / "predictions_verdict.json"))
P1 = V["P1"]["cells"]; P3 = V["P3"]["cells"]
NAME = {"qwen": "Qwen", "gemma": "Gemma"}
fails = {}
for k, ok in P1.items():
    if not ok:
        v, d, n = k.split("/")
        fails.setdefault(d, {}).setdefault(v, []).append(n)
fail_txt = "; ".join(f"{DS[d]}: " + ", ".join(f"{NAME[v]} at {' and '.join(ns)}" for v, ns in fails[d].items()) for d in fails)
held = [d for d in ("artphoto", "abstract", "memcat") if d not in fails]
p1_detail = (("holds on " + " and ".join(DS[d] for d in held) + " for both VLMs; " if held else "")
             + ("fails for " + fail_txt + r" labels (gains positive but not Holm-significant)" if fails else "holds in every cell"))
rows = [
    ("P1", r"hybrid $>$ CLIP-PCA$2k$ at 20 and 40 labels, every dataset and VLM",
     "Confirmed" if V["P1"]["confirmed"] else "Not confirmed", p1_detail),
    ("P2", r"hybrid's advantage over full CLIP shrinks with labels; not significant on MemCat at the full pool",
     "Confirmed" if V["P2"]["confirmed"] else "Not confirmed", "on every dataset and VLM"),
    ("P3", r"the VLM with the higher zero-shot score gains more at 20 labels",
     "Confirmed" if V["P3"]["confirmed"] else "Not confirmed",
     "; ".join(f"{DS[d]}: " + ", ".join(f"{NAME[v]} ${g:+.3f}$" for v, g in P3[d]["gain_at_20"].items()) for d in P3)
     + r"; every margin is small relative to its 95\% interval"),
    ("P4", r"attributes alone do not beat CLIP-PCA$k$ at 160 labels or more",
     "Confirmed" if V["P4"]["confirmed"] else "Not confirmed", r"no Holm-significant exception"),
]
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Predictions fixed before any VLM label, CLIP feature or result existed for ArtPhoto, Abstract and MemCat, and "
     r"their outcome (Qwen2.5-VL-7B and Gemma-3-4B; Holm correction over the pre-specified family of "
     r"hybrid$-$CLIP-PCA$2k$, attributes$-$CLIP-PCA$k$ and hybrid$-$CLIP contrasts).}", r"\label{tab:prereg}", r"\small",
     r"\begin{tabular}{@{}p{0.05\linewidth}p{0.35\linewidth}p{0.13\linewidth}p{0.37\linewidth}@{}}", r"\toprule",
     r"& Prediction & Outcome & Detail \\", r"\midrule"]
for a, b, c, e in rows:
    L.append(f"{a} & {b} & {c} & {e} " + r"\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(OUT / "tab_prereg.tex", "w").write("\n".join(L) + "\n")

# ---------------- figures ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
COL = {"claude": "#2a78d6", "qwen": "#eb6834", "gemma": "#1baf7a"}
MK = {"claude": "o", "qwen": "^", "gemma": "s"}

# two rows: memorability (2 panels, centred) above emotion (3 panels)
fig = plt.figure(figsize=(10.5, 7.4))
gs = fig.add_gridspec(2, 6, hspace=0.42, wspace=0.9)
slots = {"lamem": gs[0, 1:3], "memcat": gs[0, 3:5], "emoset": gs[1, 0:2], "artphoto": gs[1, 2:4], "abstract": gs[1, 4:6]}
axes = []
for d in ("lamem", "memcat", "emoset", "artphoto", "abstract"):
    ax = fig.add_subplot(slots[d]); axes.append(ax)
    vs = [v for v in VLMS if (d, v) in R]
    base = R[(d, vs[0])]["curve"]; ns = [r["n"] for r in base]
    ax.plot(ns, [r["clip"] for r in base], color="#0b0b0b", lw=2, label="CLIP (768-d)")
    ax.plot(ns, [r["pcak"] for r in base], color="#8c8b86", lw=2, ls="--", label="CLIP-PCA$k$")
    for v in vs:
        c = R[(d, v)]["curve"]
        ax.plot([r["n"] for r in c], [r["hybrid"] for r in c], color=COL[v], lw=2, marker=MK[v], ms=5.5,
                label=f"hybrid, {VLMS[v]}")
    ax.set_xscale("log"); ax.set_title(DS[d], fontsize=11); ax.set_xlabel("labelled training images", fontsize=9)
    ax.set_ylabel("accuracy" if d in EMOTION else r"Spearman $\rho$", fontsize=9)
    ticks = [t for t in (20, 80, 320, 1280) if 1.6 * t <= ns[-1]] + [ns[-1]]
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
    ax.tick_params(labelsize=8.5)
handles = {}
for ax in axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.setdefault(l, h)
fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=5, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, 0.0))
fig.text(0.5, 0.965, "Memorability", ha="center", fontsize=12, weight="bold")
fig.text(0.5, 0.505, "Emotion", ha="center", fontsize=12, weight="bold")
fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.12)
fig.savefig(FIG / "fig_curves.png", dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 3.3))
MKD = {"emoset": "o", "artphoto": "D", "abstract": "P", "lamem": "v", "memcat": "X"}
xs, ys = [], []
for (d, v) in CELLS:
    x = ZS[(d, v)]["agree"]; g = FAM[(d, v, 20, "added")]["m"]
    xs.append(x); ys.append(g)
    ax.scatter(x, g, s=46, color=COL[v], marker=MKD[d], edgecolor="#fcfcfb", linewidth=1.2, zorder=3)
    off = {("artphoto", "qwen"): (-44, 5), ("lamem", "qwen"): (6, 4), ("lamem", "gemma"): (7, -8),
           ("emoset", "qwen"): (6, -9)}.get((d, v), (5, 3))
    ax.annotate(f"{DS[d]}", (x, g), textcoords="offset points", xytext=off, fontsize=6.5, color="#52514e")
ax.axhline(0, color="#8c8b86", lw=0.8)
rho, pv = spearmanr(xs, ys)
ax.set_xlabel(r"zero-shot agreement with humans (Cohen's $\kappa$ or Spearman $\rho$)")
ax.set_ylabel("gain from adding attributes, 20 labels")
for v in VLMS:
    ax.scatter([], [], color=COL[v], marker="s", label=VLMS[v])
ax.legend(frameon=False, loc="upper left", title="colour: VLM; shape: dataset", title_fontsize=6.5)
ax.set_title(f"Spearman over {len(xs)} dataset-VLM cells: {rho:.2f} (p = {pv:.1e})", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_rule.png", dpi=300); plt.close(fig)
json.dump(dict(rho=rho, p=pv, cells=[dict(d=d, v=v, agree=x, gain20=y) for (d, v), x, y in zip(CELLS, xs, ys)]),
          open(OUT / "rule.json", "w"), indent=1)
print("rule: spearman", round(rho, 3), "p", round(pv, 4))
