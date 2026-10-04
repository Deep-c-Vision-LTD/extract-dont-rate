"""(a) Schema sensitivity: complement test with random halves of the attribute schema.
For each VLM and task, 10 random attribute subsets of size k/2; hybrid = PCA_k + subset is compared
with the equal-size code PCA_(k + k/2), at 40, 160 and 640 labels (8 training subsets each, same
protocol as analysis.py). (b) Distillation fidelity: 5-fold CV Pearson r of a ridge head from
CLIP to each VLM attribute, on all training and test images."""
import json, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.argv = [sys.argv[0], "claude"]
import analysis as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

def load(task, pfx):
    F.PFX = pfx
    if task == "emoset":
        (_, y, Xa, Xc, _), (_, yte, Xa_te, Xc_te, _), ex, _ = F.emoset()
        kt = F.dedup(Xc, Xc_te); y, Xa, Xc = y[kt], Xa[kt], Xc[kt]
        pool = ex[F.dedup(ex, np.vstack([Xc, Xc_te]))]; ncomp = 30; classif = True
    else:
        (_, y, Xa, Xc, _), (_, yte, Xa_te, Xc_te, _), pool, _ = F.lamem()
        kr = F.dedup(Xc, Xc_te); y, Xa, Xc = y[kr], Xa[kr], Xc[kr]
        pool = pool[F.dedup(pool, np.vstack([Xc, Xc_te]))]; ncomp = 24; classif = False
    pca = PCA(n_components=ncomp, random_state=0).fit(pool)
    return y, Xa, pca.transform(Xc), yte, Xa_te, pca.transform(Xc_te), classif, Xc, Xc_te

out = {}
for vlm, pfx in (("claude", ""), ("qwen", "qwen_")):
    for task in ("emoset", "lamem"):
        y, Xa, Z, yte, Xa_te, Zte, classif, Xc, Xc_te = load(task, pfx)
        k = Xa.shape[1]; h = k // 2; rs = np.random.RandomState(7)
        subsets = [sorted(rs.choice(k, h, replace=False)) for _ in range(10)]
        res = {}
        for n in (40, 160, 640):
            gains = []
            for sset in subsets:
                rng = np.random.RandomState(0); g = []
                for _ in range(8):
                    sub = (train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=rng.randint(1 << 30))[0]
                           if classif else rng.choice(len(y), n, replace=False))
                    A = np.hstack([Z[:, :k], Xa[:, sset]]); B = np.hstack([Zte[:, :k], Xa_te[:, sset]])
                    pa = F.fitpred(A[sub], y[sub], B, classif)[0]
                    pb = F.fitpred(Z[sub, :k + h], y[sub], Zte[:, :k + h], classif)[0]
                    g.append(F.metric(yte, pa, classif) - F.metric(yte, pb, classif))
                gains.append(float(np.mean(g)))
            res[n] = {"mean": round(float(np.mean(gains)), 4), "min": round(min(gains), 4), "max": round(max(gains), 4),
                      "n_positive": int(sum(x > 0 for x in gains))}
            print(vlm, task, n, res[n], flush=True)
        # distillation fidelity on all training + test images
        X = np.vstack([Xc, Xc_te]); A = np.vstack([Xa, Xa_te]); dist = {}
        names = (F.EMO_NUM + ["has_people"]) if task == "emoset" else (F.MEM_NUM + ["has_people", "is_indoor_scene"])
        for j, nm in enumerate(names):
            p = np.zeros(len(A))
            for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
                sc = StandardScaler().fit(X[tr]); p[te] = Ridge(alpha=10.0).fit(sc.transform(X[tr]), A[tr, j]).predict(sc.transform(X[te]))
            dist[nm] = round(float(pearsonr(A[:, j], p)[0]), 3)
        print(vlm, task, "distill", dist, flush=True)
        out[f"{vlm}_{task}"] = {"schema_half": res, "subsets": [list(map(int, s)) for s in subsets], "distill": dist}
json.dump(out, open(F.HERE / "results" / "schema_distill_results.json", "w"), indent=1)
print("saved schema_distill_results.json")
