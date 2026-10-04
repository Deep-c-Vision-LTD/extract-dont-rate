"""
Single source of every number in "Complement, Don't Replace" (results/final_results_<vlm>.json).
One protocol for every route, both benchmarks, one VLM at a time.

Routes (all with a linear head; regularization re-selected by inner CV on every subset):
  attrs   : VLM-extracted attribute vector (k = 10 EmoSet, 8 LaMem)
  pcak    : CLIP compressed to k dims (dimension-matched to attrs)
  pca2k   : CLIP compressed to 2k dims (equal-size control for the hybrid)
  hybrid  : pcak + attrs (2k dims)
  clip    : full 768-d CLIP
  direct  : zero-shot VLM rating (no training)
PCA is fitted on CLIP embeddings of a disjoint pool whose labels are never used
(EmoSet 20,000-image pool; LaMem 6,000-image train_1 pool).

Usage: python analysis.py claude|qwen [emoset] [lamem]   -> results/final_results_<vlm>.json
"""
import csv, json, sys, warnings
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
VLM = sys.argv[1] if len(sys.argv) > 1 else "claude"
PFX = "" if VLM == "claude" else "qwen_"
QWEN_FILES = {"emoset": {"emoset_features": 1500, "emoset2k_features": 1900},
              "lamem": {"lamemtr2k_features": 1900, "lamem2k_features": 1900}}
TASKS = sys.argv[2:] or ["emoset", "lamem"]


def _qwen_done(task):
    return all((HERE / "labels" / f"qwen_{f}_labels.csv").exists() and
               len(list(csv.DictReader(open(HERE / "labels" / f"qwen_{f}_labels.csv")))) >= n
               for f, n in QWEN_FILES[task].items())


# Evaluate each task on the images that EVERY available VLM labelled completely, so both VLMs see
# identical sets once the second VLM's labels for that task exist.
VLMS_FOR_IDS = {t: (["", "qwen_"] if _qwen_done(t) else [""]) for t in ("emoset", "lamem")}
SEED, REPS, NBOOT = 0, 8, 1000
CS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
ALPHAS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
EMO = ["amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"]
EMO_NUM = ["valence_7", "arousal_5", "has_threat_or_danger", "has_disgusting_content",
           "novelty_5", "playfulness_5", "warmth_5", "energy_5", "aesthetic_5"]
MEM_NUM = ["focal_subject_clarity_5", "distinctiveness_5", "visual_complexity_5",
           "color_vividness_5", "text_amount_5", "emotional_charge_5"]
PEOPLE = {"none": 0, "people_present": 1, "prominent_faces": 2}
INDOOR = {"indoor": 0, "outdoor": 1, "unclear": 0.5}


def rows(name):
    p = HERE / "labels" / f"{name}.csv"
    return {r["id"]: r for r in csv.DictReader(open(p))} if p.exists() else {}


def clipmap(npz):
    d = np.load(HERE / "features" / npz)
    return {n.rsplit(".", 1)[0]: e for n, e in zip(d["names"], d["emb"])}


def emo_vec(r):
    return [float(r[c]) for c in EMO_NUM] + [PEOPLE[r["has_people"]]]


def mem_vec(r):
    return [float(r[c]) for c in MEM_NUM] + [PEOPLE[r["has_people"]], INDOOR[r["is_indoor_scene"]]]


def ok(r, cols, cats):
    return r is not None and all(r.get(c, "") != "" for c in cols) and all(r.get(c) in m for c, m in cats.items())


# ---------------- data ----------------
def emoset():
    """Training pool: 2,000 images from EmoSet's training split (VLM-labelled).
    Test set: the 1,600-image validation subset (VLM-labelled, including direct rating)."""
    def part(sample, npz, fname, dname):
        gt = {Path(g["image_rel"]).stem: g["label"] for g in json.load(open(HERE / "samples" / sample))}
        cb = clipmap(npz)
        feats = {p: rows(f"{p}{fname}") for p in VLMS_FOR_IDS["emoset"]}
        ids = [i for i in gt if i in cb and all(ok(feats[p].get(i), EMO_NUM, {"has_people": PEOPLE}) for p in VLMS_FOR_IDS["emoset"])]
        f = rows(f"{PFX}{fname}")
        y = np.array([EMO.index(gt[i]) for i in ids])
        Xa, Xc = np.array([emo_vec(f[i]) for i in ids]), np.array([cb[i] for i in ids])
        direct = None
        if dname:
            d = rows(f"{PFX}{dname}")
            direct = np.array([EMO.index(d[i]["emotion"]) if i in d and d[i].get("emotion") in EMO else -1 for i in ids])
        return ids, y, Xa, Xc, direct
    tr = part("emoset_train2k_sample.json", "emoset_train2k_clip.npz", "emoset2k_features_labels", None)
    te = part("emoset_sample.json", "emoset_clip.npz", "emoset_features_labels", "emoset_direct_labels")
    exz = np.load(HERE / "features" / "emoset_extra_clip.npz")
    lab = json.load(open(HERE / "features" / "emoset_extra_labels.json"))
    ex, exl = exz["emb"], np.array([EMO.index(lab[str(n)]) for n in exz["names"]])
    return tr, te, ex, exl


def lamem():
    sc_tr = {Path(g["image"]).stem: g["score"] for g in json.load(open(HERE / "samples" / "lamem_train2k_sample.json"))}
    sc_te = {Path(g["image"]).stem: g["score"] for g in json.load(open(HERE / "samples" / "lamem_test2k_sample.json"))}
    cb_tr, cb_te = clipmap("lamem_train2k_clip.npz"), clipmap("lamem_test2k_clip.npz")
    cats = {"has_people": PEOPLE, "is_indoor_scene": INDOOR}

    def part(sc, cb, fname, dname):
        feats = {p: rows(f"{p}{fname}") for p in VLMS_FOR_IDS["lamem"]}
        ids = [i for i in sc if i in cb and all(ok(feats[p].get(i), MEM_NUM, cats) for p in VLMS_FOR_IDS["lamem"])]
        f, d = rows(f"{PFX}{fname}"), rows(f"{PFX}{dname}") if dname else {}
        y = np.array([sc[i] for i in ids]); Xa = np.array([mem_vec(f[i]) for i in ids])
        Xc = np.array([cb[i] for i in ids])
        direct = np.array([float(d[i]["memorability_1_10"]) if i in d and d[i].get("memorability_1_10", "") != ""
                           else np.nan for i in ids]) if dname else None
        return ids, y, Xa, Xc, direct

    tr = part(sc_tr, cb_tr, "lamemtr2k_features_labels", None)
    te = part(sc_te, cb_te, "lamem2k_features_labels", "lamem2k_direct_labels")
    pool = json.load(open(HERE / "features" / "lamem_train6000_scores.json"))
    pc = np.load(HERE / "features" / "lamem_train6000_clip.npz")
    keep = [j for j, n in enumerate(pc["names"]) if n in pool]
    return tr, te, pc["emb"][keep], np.array([pool[pc["names"][j]] for j in keep])


def dedup(P, ref, thr=0.95):
    """Drop pool rows whose cosine similarity to any reference image exceeds thr."""
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True); Rn = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    keep = (Pn @ Rn.T).max(1) <= thr
    return keep


# ---------------- models ----------------
def tune(X, y, classif):
    if classif:
        k = max(2, min(5, np.bincount(y).min()))
        folds = list(StratifiedKFold(k, shuffle=True, random_state=SEED).split(X, y))
    else:
        folds = list(KFold(min(5, len(y) // 4), shuffle=True, random_state=SEED).split(X))
    best, bs = None, -np.inf
    for h in (CS if classif else ALPHAS):
        p = np.zeros(len(y))
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=4000, C=h) if classif else Ridge(alpha=h)
            p[te] = m.fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
        s = (p == y).mean() if classif else spearmanr(y, p)[0]
        if s > bs:
            bs, best = s, h
    return best


def fitpred(Xtr, ytr, Xte, classif):
    h = tune(Xtr, ytr, classif)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=4000, C=h) if classif else Ridge(alpha=h)
    return m.fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte)), h


def metric(y, p, classif):
    return float((p == y).mean()) if classif else float(spearmanr(y, p)[0])


def boot(y, PA, PB, classif, rng):
    """Two-level paired bootstrap of the subset-averaged difference metric(B) - metric(A):
    each resample draws training subsets (with replacement, from the 8 drawn) AND test items,
    so the interval reflects both which images were labelled and which were tested."""
    d, k = [], len(PA)
    for _ in range(NBOOT):
        s = rng.randint(0, len(y), len(y))
        js = rng.randint(0, k, k) if k > 1 else [0]
        d.append(np.mean([metric(y[s], PB[j][s], classif) - metric(y[s], PA[j][s], classif) for j in js]))
    d = np.array(d)
    return [round(float(np.mean(d)), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


CONTRASTS = [("hybrid", "pca2k"), ("attrs", "pcak"), ("hybrid", "clip"), ("attrs", "clip"), ("clip", "pcak")]


def curve(name, Xa, Xc, y, Xa_te, Xc_te, y_te, pca, pool_size, classif, sizes, direct_te, draw):
    k = Xa.shape[1]
    Z, Zte = pca.transform(Xc), pca.transform(Xc_te)
    R = {"attrs": (Xa, Xa_te), "pcak": (Z[:, :k], Zte[:, :k]), "pca2k": (Z[:, :2 * k], Zte[:, :2 * k]),
         "hybrid": (np.hstack([Z[:, :k], Xa]), np.hstack([Zte[:, :k], Xa_te])), "clip": (Xc, Xc_te)}
    rng_b, res = np.random.RandomState(1), []
    valid = ~np.isnan(direct_te) if not classif else direct_te >= 0
    for n in sizes:
        rng = np.random.RandomState(SEED)
        P = {r: [] for r in R}; H = {r: [] for r in R}
        for _ in range(REPS if n < pool_size else 1):
            sub = draw(n, rng)
            for r, (A, B) in R.items():
                p, h = fitpred(A[sub], y[sub], B, classif)
                P[r].append(p); H[r].append(h)
        row = {"n": int(n), **{r: round(float(np.mean([metric(y_te, p, classif) for p in P[r]])), 4) for r in R}}
        row["ci"] = {f"{a}-{b}": boot(y_te, P[b], P[a], classif, rng_b) for a, b in CONTRASTS}
        # trained routes vs zero-shot direct rating, on items with a valid direct answer
        dpred = [direct_te[valid]] * len(P["attrs"])
        row["ci"].update({f"{r}-direct": boot(y_te[valid], dpred, [p[valid] for p in P[r]], classif, rng_b)
                          for r in ("attrs", "pcak", "clip", "hybrid")})
        row["hyper"] = {r: H[r] for r in ("attrs", "clip", "hybrid")}
        res.append(row)
        print(f"[{name}] n={n:5d} " + " ".join(f"{r} {row[r]:.3f}" for r in R) +
              f" | hybrid-pca2k {row['ci']['hybrid-pca2k']} attrs-pcak {row['ci']['attrs-pcak']}", flush=True)
    return res


def run_emoset(out):
    """EmoSet: train on the official training split, test on the validation subset."""
    (ids_tr, y, Xa, Xc, _), (ids_te, yte, Xa_te, Xc_te, d_te), ex, exl = emoset()
    kt = dedup(Xc, Xc_te)                      # training images that near-duplicate a test image
    ids_tr = [i for i, k in zip(ids_tr, kt) if k]; y, Xa, Xc = y[kt], Xa[kt], Xc[kt]
    kx = dedup(ex, np.vstack([Xc, Xc_te])); ex, exl = ex[kx], exl[kx]
    pca = PCA(n_components=20, random_state=SEED).fit(ex)
    ok_d = d_te >= 0
    out["emoset"] = {"n_pool": len(y), "n_test": len(yte), "pool_removed_dups": int((~kt).sum()),
                     "extra_removed_dups": int((~kx).sum()),
                     "direct_acc": metric(yte[ok_d], d_te[ok_d], True), "direct_valid": int(ok_d.sum())}
    out["emoset"]["curve"] = curve("EmoSet", Xa, Xc, y, Xa_te, Xc_te, yte, pca, len(y), True,
                                   [20, 40, 80, 160, 320, 640, 1280, len(y)], d_te,
                                   lambda n, rng: np.arange(len(y)) if n >= len(y) else
                                   train_test_split(np.arange(len(y)), train_size=n, stratify=y,
                                                    random_state=rng.randint(1 << 30))[0])
    # large-budget reference: CLIP with the label-only pool
    pb, _ = fitpred(np.vstack([Xc, ex]), np.concatenate([y, exl]), Xc_te, True)
    out["emoset"]["clip_plus_pool"] = {"acc": metric(yte, pb, True), "n_train": len(y) + len(ex)}
    print("EmoSet", {k: v for k, v in out["emoset"].items() if k != "curve"}, flush=True)


def run_lamem(out):
    """LaMem: train on train_1, test on test_1."""
    (ids_tr, ytr, Xa_tr, Xc_tr, _), (ids_te, yte, Xa_te, Xc_te, dte), Xpool, ypool = lamem()
    kr = dedup(Xc_tr, Xc_te)                   # training images that near-duplicate a test image
    ytr, Xa_tr, Xc_tr = ytr[kr], Xa_tr[kr], Xc_tr[kr]
    kx = dedup(Xpool, np.vstack([Xc_tr, Xc_te])); Xpool, ypool = Xpool[kx], ypool[kx]
    kt = np.ones(len(yte), bool)
    print(f"LaMem: removed {(~kr).sum()} training and {(~kx).sum()} pool near-duplicates", flush=True)
    ids_te = [i for i, k in zip(ids_te, kt) if k]; yte, Xa_te, Xc_te, dte = yte[kt], Xa_te[kt], Xc_te[kt], dte[kt]
    pca = PCA(n_components=16, random_state=SEED).fit(Xpool)
    ok_d = ~np.isnan(dte)
    out["lamem"] = {"n_pool": len(ytr), "n_test": len(yte), "train_removed_dups": int((~kr).sum()), "extra_removed_dups": int((~kx).sum()), "direct_rho": metric(yte[ok_d], dte[ok_d], False),
                    "direct_valid": int(ok_d.sum())}
    out["lamem"]["curve"] = curve("LaMem", Xa_tr, Xc_tr, ytr, Xa_te, Xc_te, yte, pca, len(ytr), False,
                                  [20, 40, 80, 160, 320, 640, 1280, len(ytr)], dte,
                                  lambda n, rng: np.arange(len(ytr)) if n >= len(ytr) else rng.choice(len(ytr), n, replace=False))
    pb, _ = fitpred(np.vstack([Xc_tr, Xpool]), np.concatenate([ytr, ypool]), Xc_te, False)
    out["lamem"]["clip_plus_pool"] = {"rho": metric(yte, pb, False), "n_train": len(ytr) + len(ypool)}
    # what the attributes add: Spearman of each attribute with the full-pool CLIP residual
    pc, _ = fitpred(Xc_tr, ytr, Xc_te, False)
    names = MEM_NUM + ["has_people", "is_indoor_scene"]
    out["lamem"]["residual_corr"] = {n: round(float(spearmanr(Xa_te[:, j], yte - pc)[0]), 3) for j, n in enumerate(names)}
    print("LaMem", {k: v for k, v in out["lamem"].items() if k != "curve"}, flush=True)


def main():
    out_path = HERE / "results" / f"final_results_{VLM}.json"
    out = json.load(open(out_path)) if out_path.exists() else {}
    out.update({"vlm": VLM, "ids_from": VLMS_FOR_IDS})
    if "emoset" in TASKS:
        run_emoset(out)
    if "lamem" in TASKS:
        run_lamem(out)
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"saved {out_path.name} (tasks run: {TASKS})")


if __name__ == "__main__":
    main()
