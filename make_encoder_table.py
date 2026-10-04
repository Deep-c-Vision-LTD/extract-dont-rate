"""Table of encoder dependence, generated from the result files (no hand-copied numbers)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ENC = [("clip", "CLIP L/14"), ("siglipl", "SigLIP-L"), ("dinov2l", "DINOv2-L")]
VLM = [("claude", "Claude"), ("qwen", "Qwen")]
TASK = [("emoset", "EmoSet"), ("lamem", "LaMem")]
dist = json.load(open(HERE / "results" / "encoder_distill_results.json"))

def load(enc, v):
    p = HERE / "results" / (f"final_results_{v}.json" if enc == "clip" else f"encoder_robustness_{enc}_{v}.json")
    return json.load(open(p))

def budgets(curve, key, sign):
    ns = []
    for i, r in enumerate(curve):
        lo, hi = r["ci"][key][1], r["ci"][key][2]
        if (sign > 0 and lo > 0) or (sign < 0 and hi < 0):
            ns.append("full" if i == len(curve) - 1 else str(r["n"]))
    if not ns:
        return "none"
    allns = ["full" if i == len(curve) - 1 else str(r["n"]) for i, r in enumerate(curve)]
    if len(ns) == len(allns):
        return "all"
    idx = [allns.index(x) for x in ns]
    if idx == list(range(idx[0], idx[-1] + 1)):
        return ns[0] if len(ns) == 1 else f"{ns[0]}--{ns[-1]}"
    return ", ".join(ns)

L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Dependence on the frozen encoder. \emph{Recov.}: mean Pearson $r$ with which the encoder "
     r"recovers the VLM's attributes (5-fold ridge). \emph{Complement}: label budgets at which the hybrid "
     r"significantly beats the equal-size code PCA$2k$. \emph{Substitute}: budgets at which the attributes "
     r"significantly beat (+) or trail ($-$) the dimension-matched code PCA$k$. \emph{Full}: the full "
     r"encoder probe with the full pool (accuracy or Spearman $\rho$).}",
     r"\label{tab:encoders}", r"\footnotesize\setlength{\tabcolsep}{3pt}",
     r"\begin{tabular}{@{}lllccccc@{}}", r"\toprule",
     r"Encoder & VLM & Task & Recov. & Complement & Subst.\\ (+) & Subst.\\ ($-$) & Full \\", r"\midrule"]
for enc, en in ENC:
    for vi, (v, vn) in enumerate(VLM):
        R = load(enc, v)
        for ti, (t, tn) in enumerate(TASK):
            c = R[t]["curve"]
            rec = dist[f"{v}_{enc}"][f"{t}_mean_r"]
            full = c[-1]["clip"]
            head = (en if vi == 0 and ti == 0 else "") + " & " + (vn if ti == 0 else "") + f" & {tn}"
            L.append(f"{head} & {rec:.2f} & {budgets(c, 'hybrid-pca2k', 1)} & {budgets(c, 'attrs-pcak', 1)} & "
                     f"{budgets(c, 'attrs-pcak', -1)} & {full:.3f} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}", r"\end{table}"]
(HERE / "tables").mkdir(exist_ok=True); (HERE / "tables" / "tab_encoders.tex").write_text("\n".join(L) + "\n")
print("\n".join(L))
