"""
Builds the paper's tables and figure from results/final_results_<vlm>.json, so no number is
transcribed by hand. Writes tables/tab_curves.tex, tables/tab_contrasts.tex, figures/fig_curves.png.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "tables"; OUT.mkdir(exist_ok=True)
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
R = {v: json.load(open(HERE / "results" / f"final_results_{v}.json"))
     for v in ("claude", "qwen") if (HERE / "results" / f"final_results_{v}.json").exists()}
NAME = {"claude": "Claude Sonnet 4.6", "qwen": "Qwen2.5-VL-7B"}
TASK = {"emoset": "EmoSet (accuracy)", "lamem": r"LaMem (Spearman $\rho$)"}


def f3(x):
    return f"{x:.3f}"


def sig(ci):
    return ci[1] > 0 or ci[2] < 0


def delta(ci):
    s = f"{ci[0]:+.3f}".replace("-", "-")
    return rf"$\mathbf{{{s}}}$" if sig(ci) else f"${s}$"


# ---- Table: learning curves for every route (one block per VLM) ----
def tab_curves():
    L = [r"\begin{table*}[p]", r"\centering",
         r"\caption{Test performance of each route against the number of labelled training images "
         r"(mean of 8 training subsets; regularization re-selected on every subset). $k$ = number of "
         r"VLM attributes (10 EmoSet, 8 LaMem). Bold: highest mean at that size; Table~\ref{tab:contrasts} "
         r"gives the intervals. The last row of each block is the full pool. Direct rating needs no labels.}",
         r"\label{tab:curves}", r"\small\setlength{\tabcolsep}{4pt}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrccccc}", r"\toprule",
         r"VLM & Task & $n$ & Attributes & CLIP-PCA$k$ & CLIP (768-d) & Hybrid & Direct \\",
         r"\midrule"]
    for v, res in R.items():
        for t in ("emoset", "lamem"):
            d = res[t]["direct_acc"] if t == "emoset" else res[t]["direct_rho"]
            for j, row in enumerate(res[t]["curve"]):
                vals = {r: row[r] for r in ("attrs", "pcak", "clip", "hybrid")}
                best = max(vals, key=vals.get)
                cells = [rf"\textbf{{{f3(vals[r])}}}" if r == best else f3(vals[r]) for r in vals]
                n = f"full ({row['n']})" if j == len(res[t]["curve"]) - 1 else str(row["n"])
                head = (NAME[v] if t == "emoset" and j == 0 else "") + " & " + (TASK[t] if j == 0 else "")
                L.append(f"{head} & {n} & " + " & ".join(cells) + f" & {f3(d) if j == 0 else ''} \\\\")
            L.append(r"\addlinespace")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}}", r"\end{table*}"]
    (OUT / "tab_curves.tex").write_text("\n".join(L) + "\n")


# ---- Table: paired-bootstrap contrasts ----
def tab_contrasts():
    C = [("hybrid-pca2k", r"Hybrid $-$ PCA$2k$"), ("attrs-pcak", r"Attributes $-$ PCA$k$"),
         ("hybrid-clip", r"Hybrid $-$ CLIP")]
    ns = [row["n"] for row in next(iter(R.values()))["emoset"]["curve"]]
    L = [r"\begin{table*}[p]", r"\centering",
         r"\caption{Differences between routes with two-level paired bootstrap 95\% intervals (1,000 "
         r"resamples of both the training subsets and the test images). Hybrid $-$ PCA$2k$ compares "
         r"codes of equal size that differ only in whether half of the dimensions are VLM attributes "
         r"(complement test); Attributes $-$ PCA$k$ compares the VLM attributes with a "
         r"dimension-matched, label-free compression of CLIP (substitute test). EmoSet differences are "
         r"in accuracy, LaMem differences in Spearman $\rho$. Bold: the interval excludes zero.}",
         r"\label{tab:contrasts}", r"\footnotesize\setlength{\tabcolsep}{1.8pt}",
         r"\begin{tabular}{@{}l" + "r" * len(ns) + "@{}}", r"\toprule",
         r"Contrast \textbackslash{} $n$ & " + " & ".join(["full" if i == len(ns) - 1 else str(n) for i, n in enumerate(ns)]) + r" \\",
         r"\midrule"]
    for v, res in R.items():
        for t in ("emoset", "lamem"):
            L.append(rf"\multicolumn{{{len(ns) + 1}}}{{@{{}}l}}{{\textit{{{NAME[v]}, {dict(emoset='EmoSet', lamem='LaMem')[t]}}}}} \\")
            for key, lab in C:
                L.append(f"{lab} & " + " & ".join(delta(row["ci"][key]) for row in res[t]["curve"]) + r" \\")
            L.append(r"\addlinespace")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table*}"]
    (OUT / "tab_contrasts.tex").write_text("\n".join(L) + "\n")


# ---- Figure: curves (Claude solid, Qwen dashed) ----
def fig_curves():
    col = {"attrs": "#d1495b", "pcak": "#edae49", "clip": "#2e5090", "hybrid": "#00798c"}
    lab = {"attrs": "VLM attributes", "pcak": "CLIP-PCA$k$ (label-free compression)",
           "clip": "CLIP, 768-d", "hybrid": "Hybrid, PCA$k$ + attributes:"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, t, yl in [(axes[0], "emoset", "Accuracy"), (axes[1], "lamem", r"Spearman $\rho$")]:
        for v, ls, mk in [("claude", "-", "o"), ("qwen", "--", "^")]:
            if v not in R:
                continue
            cur = R[v][t]["curve"]; ns = [r["n"] for r in cur]
            routes = ("attrs", "pcak", "clip", "hybrid") if v == "claude" else ("attrs", "hybrid")
            for r in routes:
                ax.plot(ns, [x[r] for x in cur], ls, color=col[r], marker=mk, ms=4, lw=1.8,
                        label=(f"{lab[r].split(':')[0]} ({NAME[v]})" if r in ("attrs", "hybrid") else lab[r]))
            d = R[v][t]["direct_acc"] if t == "emoset" else R[v][t]["direct_rho"]
            ax.axhline(d, color="#555555", ls=":" if v == "qwen" else "--", lw=1.1,
                       label=f"Direct rating ({NAME[v]})")
        ns = [r["n"] for r in R["claude"][t]["curve"]]
        tk = [n for i, n in enumerate(ns) if not (i == len(ns) - 2 and ns[-1] < 1.7 * n)]
        ax.set_xscale("log"); ax.set_xticks(tk); ax.set_xticklabels([str(n) for n in tk]); ax.minorticks_off()
        ax.set_xlabel("Labelled training images (log scale)"); ax.set_ylabel(yl)
        ax.set_title({"emoset": "EmoSet, 8-way emotion", "lamem": "LaMem, memorability (2,000 test images)"}[t], fontsize=10)
        ax.grid(alpha=0.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=[0, 0.17 if "qwen" in R else 0.12, 1, 1])
    fig.savefig(FIG / "fig_curves.png", dpi=250)


tab_curves(); tab_contrasts(); fig_curves()
print("wrote tables/tab_curves.tex, tables/tab_contrasts.tex, figures/fig_curves.png")
