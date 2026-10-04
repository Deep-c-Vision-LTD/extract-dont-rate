"""Encoder robustness: rerun the main analysis with DINOv2-L or SigLIP-L in place of CLIP ViT-L/14.
Images, splits, VLM attributes, near-duplicate removal (computed on CLIP, so the image sets are
identical), budgets, tuning, and bootstrap are exactly those of analysis.py; only the frozen
encoder (and therefore the PCA compression) changes.
Usage: python encoder_robustness.py <dinov2l|siglipl> <claude|qwen>"""
import json, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
ENC, V = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], V]
import analysis as F
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def emap(name):
    d = np.load(F.HERE / "features" / f"{name}_{ENC}.npz")
    return {n.rsplit(".", 1)[0]: e for n, e in zip(d["names"], d["emb"])}

out = {"encoder": ENC, "vlm": V}
# EmoSet
(ids_tr, y, Xa, Xc, _), (ids_te, yte, Xa_te, Xc_te, d_te), ex, exl = F.emoset()
kt = F.dedup(Xc, Xc_te); ids_tr = [i for i, k in zip(ids_tr, kt) if k]; y, Xa, Xc = y[kt], Xa[kt], Xc[kt]
kx = F.dedup(ex, np.vstack([Xc, Xc_te]))
mtr, mte = emap("emoset_train2k"), emap("emoset")
E, E_te = np.array([mtr[i] for i in ids_tr]), np.array([mte[i] for i in ids_te])
E_ex = np.load(F.HERE / "features" / f"emoset_extra_{ENC}.npz")["emb"][kx]
pca = PCA(n_components=20, random_state=F.SEED).fit(E_ex)
out["emoset"] = {"n_pool": len(y), "n_test": len(yte), "dim": E.shape[1]}
out["emoset"]["curve"] = F.curve(f"EmoSet/{ENC}/{V}", Xa, E, y, Xa_te, E_te, yte, pca, len(y), True,
                                 [20, 40, 80, 160, 320, 640, 1280, len(y)], d_te,
                                 lambda n, rng: np.arange(len(y)) if n >= len(y) else
                                 train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=rng.randint(1 << 30))[0])
# LaMem
(ids_tr, ytr, Xa_tr, Xc_tr, _), (ids_te, yte, Xa_te, Xc_te, dte), Xpool, ypool = F.lamem()
kr = F.dedup(Xc_tr, Xc_te); ids_tr = [i for i, k in zip(ids_tr, kr) if k]; ytr, Xa_tr, Xc_tr = ytr[kr], Xa_tr[kr], Xc_tr[kr]
kx = F.dedup(Xpool, np.vstack([Xc_tr, Xc_te]))
mtr, mte = emap("lamem_train2k"), emap("lamem_test2k")
E, E_te = np.array([mtr[i] for i in ids_tr]), np.array([mte[i] for i in ids_te])
pool_scores = json.load(open(F.HERE / "features" / "lamem_train6000_scores.json"))
pz = np.load(F.HERE / "features" / f"lamem_train6000_{ENC}.npz")
E_pool = pz["emb"][[j for j, n in enumerate(pz["names"]) if n in pool_scores]][kx]
pca = PCA(n_components=16, random_state=F.SEED).fit(E_pool)
out["lamem"] = {"n_pool": len(ytr), "n_test": len(yte), "dim": E.shape[1]}
out["lamem"]["curve"] = F.curve(f"LaMem/{ENC}/{V}", Xa_tr, E, ytr, Xa_te, E_te, yte, pca, len(ytr), False,
                                [20, 40, 80, 160, 320, 640, 1280, len(ytr)], dte,
                                lambda n, rng: np.arange(len(ytr)) if n >= len(ytr) else rng.choice(len(ytr), n, replace=False))
json.dump(out, open(F.HERE / "results" / f"encoder_robustness_{ENC}_{V}.json", "w"), indent=1)
print("saved", f"encoder_robustness_{ENC}_{V}.json")
