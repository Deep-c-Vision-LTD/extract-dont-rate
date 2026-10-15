"""
New datasets for the CVIU resubmission (splits fixed in PREREGISTRATION_new_datasets.md, seed 2026):
  MemCat   : 10,000 scored images -> pool 2,000 + test 2,000 (random); the remaining 6,000 are the unlabelled
             PCA pool (their scores are never used).
  ArtPhoto : 806 images, label = file-name prefix (8 classes as EmoSet) -> stratified 70/30 pool/test.
  Abstract : vote counts per class; label = unique majority, tied maxima excluded -> stratified 70/30.
Writes samples/ (image lists; MemCat scores and Abstract vote counts are not written, they stay in the official files)
and, with "clip", CLIP ViT-L/14 embeddings (same preprocessing as extract_features.py) to features/.
Usage: NEW_DATA=/path/to/cviu_new_datasets python prep_new_datasets.py [clip]
"""
import csv, json, os, re, sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent
# EDIT ME: folder holding artphoto/testImages_artphoto, abstract/testImages_abstract (with ABSTRACT_groundTruth.csv),
# memcat/MemCat (images) and memcat/memcat_image_data.csv, downloaded from the datasets' official pages
DATA = Path(os.environ.get("NEW_DATA", "/path/to/cviu_new_datasets"))
SEED = 2026
CLASSES = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]


def memcat():
    rows = list(csv.DictReader(open(DATA / "memcat" / "memcat_image_data.csv")))
    items = [{"image": f"{r['category']}/{r['subcategory']}/{r['image_file']}",
              "score": float(r["memorability_w_fa_correction"])} for r in rows]
    items = [it for it in items if (DATA / "memcat" / "MemCat" / it["image"]).exists()]
    rng = np.random.RandomState(SEED); idx = rng.permutation(len(items))
    pool, test, rest = [items[i] for i in idx[:2000]], [items[i] for i in idx[2000:4000]], [items[i] for i in idx[4000:]]
    return pool, test, rest


def artphoto():
    d = DATA / "artphoto" / "testImages_artphoto"
    items = []
    for f in sorted(os.listdir(d)):
        if f.lower().endswith(".jpg"):
            lab = f.split("_")[0].lower().replace("sad", "sadness") if f.startswith("sad") else f.split("_")[0].lower()
            items.append({"image": f, "label": lab})
    return items


def abstract():
    d = DATA / "abstract" / "testImages_abstract"
    txt = open(d / "ABSTRACT_groundTruth.csv", newline="").read().replace("\r", "\n")
    lines = [l for l in txt.split("\n") if l.strip()][1:]
    items, ties = [], 0
    for l in lines:
        parts = [p.strip().strip("'") for p in l.split(",")]
        votes = np.array([int(v) for v in parts[1:9]])
        top = np.flatnonzero(votes == votes.max())
        if len(top) > 1 or votes.max() == 0:
            ties += 1; continue
        if (d / parts[0]).exists():
            items.append({"image": parts[0], "label": CLASSES[top[0]], "votes": votes.tolist()})
    print(f"Abstract: {len(items)} with a unique majority, {ties} tied or empty excluded")
    return items


def split(items):
    y = [it["label"] for it in items]
    tr, te = train_test_split(np.arange(len(items)), test_size=0.3, stratify=y, random_state=SEED)
    return [items[i] for i in sorted(tr)], [items[i] for i in sorted(te)]


def clip_extract(paths, out):
    import torch
    from PIL import Image
    from transformers import CLIPVisionModelWithProjection
    M = np.array([0.48145466, 0.4578275, 0.40821073], np.float32); S = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
    clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval()
    prep = lambda p: ((np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR), np.float32) / 255 - M) / S).transpose(2, 0, 1)
    names, embs = [], []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            b = paths[i:i + 32]
            e = clip(pixel_values=torch.from_numpy(np.stack([prep(p) for p in b]))).image_embeds.numpy()
            embs.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)); names += [os.path.basename(p) for p in b]
    np.savez(out, names=np.array(names), emb=np.vstack(embs)); print(f"saved {out.name}: {len(names)}", flush=True)


if __name__ == "__main__":
    S_, F = REPO / "samples", REPO / "features"
    mp, mt, mr = memcat()
    ap, at = split(artphoto())
    bp, bt = split(abstract())
    sets = {"memcat_pool": mp, "memcat_test": mt, "memcat_rest": mr, "artphoto_pool": ap, "artphoto_test": at,
            "abstract_pool": bp, "abstract_test": bt}
    for name, (p, t) in (("artphoto", (ap, at)), ("abstract", (bp, bt)), ("memcat", (mp, mt))):
        sets[f"{name}_all"] = p + t
    strip = lambda k, it: {"image": it["image"]} if k.startswith("memcat") else {"image": it["image"], "label": it["label"]}
    for k, v in sets.items():
        json.dump([strip(k, it) for it in v], open(S_ / f"{k}_sample.json", "w"), indent=0)
        print(k, len(v), {c: sum(it.get("label") == c for it in v) for c in CLASSES} if "label" in v[0] else "")
    if "clip" in sys.argv:
        base = {"memcat": DATA / "memcat" / "MemCat", "artphoto": DATA / "artphoto" / "testImages_artphoto",
                "abstract": DATA / "abstract" / "testImages_abstract"}
        for k, v in sets.items():
            if not k.endswith("_all"):
                clip_extract([str(base[k.split("_")[0]] / it["image"]) for it in v], F / f"{k}_clip.npz")
