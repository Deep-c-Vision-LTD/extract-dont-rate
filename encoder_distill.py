"""How much of each VLM attribute does each frozen encoder already encode? 5-fold CV Pearson r of a
ridge head from the encoder embedding to each attribute, over all training and test images
(same images as the main analysis). Reports the mean over attributes per encoder, VLM, and task."""
import json, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.argv = [sys.argv[0], "claude"]
import analysis as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

def cv_r(X, a):
    p = np.zeros(len(a))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        sc = StandardScaler().fit(X[tr]); p[te] = Ridge(alpha=10.0).fit(sc.transform(X[tr]), a[tr]).predict(sc.transform(X[te]))
    return pearsonr(a, p)[0]

def emap(name, enc):
    d = np.load(F.HERE / "features" / (f"{name}_clip.npz" if enc == "clip" else f"{name}_{enc}.npz"))
    return {n.rsplit(".", 1)[0]: e for n, e in zip(d["names"], d["emb"])}

out = {}
for vlm, pfx in (("claude", ""), ("qwen", "qwen_")):
    F.PFX = pfx
    (ids_tr, _, Xa, _, _), (ids_te, _, Xa_te, _, _), _, _ = F.emoset()
    (lid_tr, _, La, _, _), (lid_te, _, La_te, _, _), _, _ = F.lamem()
    for enc in ("clip", "siglipl", "dinov2l"):
        e1, e2 = emap("emoset_train2k", enc), emap("emoset", enc)
        X = np.array([e1[i] for i in ids_tr] + [e2[i] for i in ids_te]); A = np.vstack([Xa, Xa_te])
        re = [cv_r(X, A[:, j]) for j in range(A.shape[1])]
        l1, l2 = emap("lamem_train2k", enc), emap("lamem_test2k", enc)
        X = np.array([l1[i] for i in lid_tr] + [l2[i] for i in lid_te]); A = np.vstack([La, La_te])
        rl = [cv_r(X, A[:, j]) for j in range(A.shape[1])]
        out[f"{vlm}_{enc}"] = {"emoset_mean_r": round(float(np.mean(re)), 3), "emoset_range": [round(min(re), 3), round(max(re), 3)],
                               "lamem_mean_r": round(float(np.mean(rl)), 3), "lamem_range": [round(min(rl), 3), round(max(rl), 3)]}
        print(vlm, enc, out[f"{vlm}_{enc}"], flush=True)
json.dump(out, open(F.HERE / "results" / "encoder_distill_results.json", "w"), indent=1)
