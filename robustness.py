"""Robustness numbers under the exact analysis.py protocol (same images, splits, dedup,
disjoint-pool PCA): (a) one-hidden-layer MLP head, (b) a single fixed regularizer, (c) drop-one attribute ablation.
Usage: python robustness.py claude|qwen   -> results/robustness_final_<vlm>.json"""
import json, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "claude"]
import analysis as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
ALPH = [1e-3, 1e-2, 1e-1, 1.0]

def mlp(classif, a, s):
    kw = dict(hidden_layer_sizes=(256,), alpha=a, max_iter=500, random_state=s)
    return MLPClassifier(**kw) if classif else MLPRegressor(**kw)

def mlp_fitpred(A, y, B, classif, s):
    sc = StandardScaler().fit(A); A, B = sc.transform(A), sc.transform(B)
    ys = y if classif else (y - y.mean()) / y.std()
    folds = list((StratifiedKFold(3, shuffle=True, random_state=0).split(A, ys)) if classif
                 else KFold(3, shuffle=True, random_state=0).split(A))
    best = max(ALPH, key=lambda a: np.mean([F.metric(ys[te], mlp(classif, a, s).fit(A[tr], ys[tr]).predict(A[te]), classif) for tr, te in folds]))
    return mlp(classif, best, s).fit(A, ys).predict(B)

def fixed_fitpred(A, y, B, classif, h):
    sc = StandardScaler().fit(A)
    m = LogisticRegression(max_iter=4000, C=h) if classif else Ridge(alpha=h)
    return m.fit(sc.transform(A), y).predict(sc.transform(B))

out = {}
(_, y, Xa, Xc, _), (_, yte, Xa_te, Xc_te, _), ex, exl = F.emoset()
kt = F.dedup(Xc, Xc_te); y, Xa, Xc = y[kt], Xa[kt], Xc[kt]
kx = F.dedup(ex, np.vstack([Xc, Xc_te])); ex = ex[kx]
pca = PCA(n_components=20, random_state=0).fit(ex); Z, Zte = pca.transform(Xc), pca.transform(Xc_te)
R = {"attrs": (Xa, Xa_te), "pcak": (Z[:, :10], Zte[:, :10]), "clip": (Xc, Xc_te),
     "hybrid": (np.hstack([Z[:, :10], Xa]), np.hstack([Zte[:, :10], Xa_te]))}
rows = []
for n in [80, 320, len(y)]:
    rng = np.random.RandomState(0); acc = {r: [] for r in R}
    for rep in range(4 if n < len(y) else 1):
        sub = np.arange(len(y)) if n >= len(y) else train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=rng.randint(1 << 30))[0]
        for r, (A, B) in R.items(): acc[r].append(F.metric(yte, mlp_fitpred(A[sub], y[sub], B, True, rep), True))
    rows.append({"n": int(n), **{r: round(float(np.mean(v)), 4) for r, v in acc.items()}}); print("EmoSet MLP", rows[-1], flush=True)
# drop-one attribute ablation of the full-pool hybrid (linear head)
A, B = R["hybrid"]; base = F.metric(yte, F.fitpred(A, y, B, True)[0], True); drop = {}
for j, nm in enumerate(F.EMO_NUM + ["has_people"]):
    keep = [c for c in range(A.shape[1]) if c != 10 + j]
    drop[nm] = round(F.metric(yte, F.fitpred(A[:, keep], y, B[:, keep], True)[0], True) - base, 4)
print("EmoSet drop-one", base, drop, flush=True); out["emoset_drop_one"] = {"base": base, **drop}
out["emoset_mlp"] = rows
(_, ytr, Xa_tr, Xc_tr, _), (ids_te, yte, Xa_te, Xc_te, dte), Xpool, ypool = F.lamem()
kx = F.dedup(Xpool, np.vstack([Xc_tr, Xc_te])); Xpool = Xpool[kx]
kr = F.dedup(Xc_tr, Xc_te); ytr, Xa_tr, Xc_tr = ytr[kr], Xa_tr[kr], Xc_tr[kr]
pca = PCA(n_components=16, random_state=0).fit(Xpool); Ztr, Zte = pca.transform(Xc_tr), pca.transform(Xc_te)
R = {"attrs": (Xa_tr, Xa_te), "pcak": (Ztr[:, :8], Zte[:, :8]), "clip": (Xc_tr, Xc_te),
     "hybrid": (np.hstack([Ztr[:, :8], Xa_tr]), np.hstack([Zte[:, :8], Xa_te]))}
rows = []
for n in [80, 320, len(ytr)]:
    rng = np.random.RandomState(0); rr = {r: [] for r in R}
    for rep in range(4 if n < len(ytr) else 1):
        sub = np.arange(len(ytr)) if n >= len(ytr) else rng.choice(len(ytr), n, replace=False)
        for r, (A, B) in R.items(): rr[r].append(F.metric(yte, mlp_fitpred(A[sub], ytr[sub], B, False, rep), False))
    rows.append({"n": n, **{r: round(float(np.mean(v)), 4) for r, v in rr.items()}}); print("LaMem MLP", rows[-1], flush=True)
out["lamem_mlp"] = rows
# fixed regularizer (alpha = 100 for every route and size)
rows = []
for n in [80, 320, len(ytr)]:
    rng = np.random.RandomState(0); rr = {r: [] for r in R}
    for rep in range(8 if n < len(ytr) else 1):
        sub = np.arange(len(ytr)) if n >= len(ytr) else rng.choice(len(ytr), n, replace=False)
        for r, (A, B) in R.items(): rr[r].append(F.metric(yte, fixed_fitpred(A[sub], ytr[sub], B, False, 100.0), False))
    rows.append({"n": n, **{r: round(float(np.mean(v)), 4) for r, v in rr.items()}}); print("LaMem fixed alpha=100", rows[-1], flush=True)
out["lamem_fixed_alpha100"] = rows
json.dump(out, open(F.HERE / "results" / f"robustness_final_{F.VLM}.json", "w"), indent=1)
