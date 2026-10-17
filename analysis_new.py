"""
New datasets (PREREGISTRATION_new_datasets.md): ArtPhoto and Abstract (8 EmoSet emotion classes, EmoSet emotion schema)
and MemCat (memorability, LaMem schema). Same routes, probe, 30 subsets, bootstrap and contrasts as analysis.py.
PCA: EmoSet 20,000-image unlabelled pool for ArtPhoto/Abstract, the 6,000 MemCat images outside pool and test for MemCat.
Near-duplicates (CLIP cosine > 0.95) are removed as in analysis.py: training images that match a test image, and PCA-pool
images that match a training or test image.
Ids = images that every VLM given in VLMS labelled completely.
MemCat scores are not redistributed: set MEMCAT_CSV to your copy of memcat_image_data.csv (MemCat OSF project).
Usage: MEMCAT_CSV=/path/to/memcat_image_data.csv VLMS=qwen,gemma,qwen35 REPS=30 python analysis_new.py qwen|gemma|qwen35
       [artphoto] [abstract] [memcat]   -> results/new_<vlm>_<probe>_r<reps>.json
"""
import json, os, sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

VLM = sys.argv[1]; SETS = sys.argv[2:] or ["artphoto", "abstract", "memcat"]
sys.argv = [sys.argv[0], VLM]
import analysis as A                                             # routes, probe, bootstrap, contrasts

VLMS = os.environ.get("VLMS", "qwen,gemma,qwen35").split(",")
BUDGETS = [20, 40, 80, 160, 320, 640, 1280]


def common_ids(job, sample, key, cols, cats):
    rows = {v: A.rows(f"{v}_{job}_labels") for v in VLMS}
    items = json.load(open(A.HERE / "samples" / sample))
    return [it for it in items if all(A.ok(rows[v].get(Path(it[key]).stem), cols, cats) for v in VLMS)], rows[VLM]


def load_emotion(name):
    cb = A.clipmap(f"{name}_pool_clip.npz") | A.clipmap(f"{name}_test_clip.npz")
    feats_ok, f = common_ids(f"{name}_features", f"{name}_all_sample.json", "image", A.EMO_NUM, {"has_people": A.PEOPLE})
    ok = {Path(it["image"]).stem for it in feats_ok}
    d = A.rows(f"{VLM}_{name}_direct_labels")
    out = []
    for part in ("pool", "test"):
        items = [it for it in json.load(open(A.HERE / "samples" / f"{name}_{part}_sample.json")) if Path(it["image"]).stem in ok]
        ids = [Path(it["image"]).stem for it in items]
        y = np.array([A.EMO.index(it["label"]) for it in items])
        Xa = np.array([A.emo_vec(f[i]) for i in ids]); Xc = np.array([cb[i] for i in ids])
        direct = np.array([A.EMO.index(d[i]["emotion"]) if i in d and d[i].get("emotion") in A.EMO else -1 for i in ids])
        out.append((ids, y, Xa, Xc, direct))
    return out


def memcat_scores():
    """Image path (category/subcategory/file) -> memorability corrected for false alarms, from the official CSV."""
    import csv
    rows = csv.DictReader(open(os.environ["MEMCAT_CSV"]))
    return {f"{r['category']}/{r['subcategory']}/{r['image_file']}": float(r["memorability_w_fa_correction"]) for r in rows}


def load_memcat():
    score = memcat_scores()
    cb = A.clipmap("memcat_pool_clip.npz") | A.clipmap("memcat_test_clip.npz")
    cats = {"has_people": A.PEOPLE, "is_indoor_scene": A.INDOOR}
    feats_ok, f = common_ids("memcat_features", "memcat_all_sample.json", "image", A.MEM_NUM, cats)
    ok = {Path(it["image"]).stem for it in feats_ok}
    d = A.rows(f"{VLM}_memcat_direct_labels")
    out = []
    for part in ("pool", "test"):
        items = [it for it in json.load(open(A.HERE / "samples" / f"memcat_{part}_sample.json")) if Path(it["image"]).stem in ok]
        ids = [Path(it["image"]).stem for it in items]
        y = np.array([score[it["image"]] for it in items])
        Xa = np.array([A.mem_vec(f[i]) for i in ids]); Xc = np.array([cb[i] for i in ids])
        direct = np.array([float(d[i]["memorability_1_10"]) if i in d and d[i].get("memorability_1_10", "") != "" else np.nan
                           for i in ids])
        out.append((ids, y, Xa, Xc, direct))
    return out


def stratified_draw(y):
    def draw(n, rng):
        if n >= len(y):
            return np.arange(len(y))
        try:
            return train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=rng.randint(1 << 30))[0]
        except ValueError:                                         # a class too rare to stratify at this budget
            return rng.choice(len(y), n, replace=False)
    return draw


def run_emotion(name, out):
    (_, y, Xa, Xc, _), (_, yte, Xa_te, Xc_te, d_te) = load_emotion(name)
    kt = A.dedup(Xc, Xc_te)                         # training images that near-duplicate a test image, as for EmoSet
    y, Xa, Xc = y[kt], Xa[kt], Xc[kt]
    ex = np.load(A.HERE / "features" / "emoset_extra_clip.npz")["emb"]
    kx = A.dedup(ex, np.vstack([Xc, Xc_te]))
    pca = PCA(n_components=20, random_state=A.SEED).fit(ex[kx])
    sizes = [b for b in BUDGETS if b < len(y)] + [len(y)]
    ok_d = d_te >= 0
    out[name] = {"n_pool": len(y), "n_test": len(yte), "pool_removed_dups": int((~kt).sum()),
                 "pca_pool_removed_dups": int((~kx).sum()), "direct_acc": A.metric(yte[ok_d], d_te[ok_d], True),
                 "direct_valid": int(ok_d.sum()), "vlms_for_ids": VLMS}
    out[name]["curve"] = A.curve(name, Xa, Xc, y, Xa_te, Xc_te, yte, pca, len(y), True, sizes, d_te, stratified_draw(y))


def run_memcat(out):
    (_, y, Xa, Xc, _), (_, yte, Xa_te, Xc_te, d_te) = load_memcat()
    kt = A.dedup(Xc, Xc_te)                         # training images that near-duplicate a test image, as for LaMem
    y, Xa, Xc = y[kt], Xa[kt], Xc[kt]
    rest = np.load(A.HERE / "features" / "memcat_rest_clip.npz")["emb"]
    kx = A.dedup(rest, np.vstack([Xc, Xc_te]))
    pca = PCA(n_components=16, random_state=A.SEED).fit(rest[kx])
    sizes = [b for b in BUDGETS if b < len(y)] + [len(y)]
    ok_d = ~np.isnan(d_te)
    out["memcat"] = {"n_pool": len(y), "n_test": len(yte), "pool_removed_dups": int((~kt).sum()),
                     "pca_pool_removed_dups": int((~kx).sum()), "direct_rho": A.metric(yte[ok_d], d_te[ok_d], False),
                     "direct_valid": int(ok_d.sum()), "vlms_for_ids": VLMS}
    out["memcat"]["curve"] = A.curve("MemCat", Xa, Xc, y, Xa_te, Xc_te, yte, pca, len(y), False, sizes, d_te,
                                     lambda n, rng: np.arange(len(y)) if n >= len(y) else rng.choice(len(y), n, replace=False))


if __name__ == "__main__":
    if "memcat" in SETS and "MEMCAT_CSV" not in os.environ:
        sys.exit("set MEMCAT_CSV to memcat_image_data.csv from the MemCat OSF project (scores are not redistributed)")
    A.OUT.mkdir(exist_ok=True)
    path = A.OUT / f"new_{VLM}_{A.PROBE}_r{A.REPS}.json"
    out = json.load(open(path)) if path.exists() else {}
    out.update({"vlm": VLM, "probe": A.PROBE, "reps": A.REPS, "nboot": A.NBOOT})
    for s in SETS:
        run_memcat(out) if s == "memcat" else run_emotion(s, out)
        print(s, {k: v for k, v in out[s].items() if k != "curve"}, flush=True)
    json.dump(out, open(path, "w"), indent=1)
    print("saved", path.name)
