"""
Recreate the image lists in samples/ (already included; only needed to verify them).

  emoset_sample.json          test set: 1,600-image class-balanced subset of EmoSet's validation split
                              (the original subset, listed as released)
  emoset_train2k_sample.json  training pool: 2,000 images (250 per class) from EmoSet's training split,
                              disjoint from the 20,000-image CLIP-only pool (features/emoset_extra_labels.json)
  lamem_train2k_sample.json   training pool: 2,000 images, uniformly at random from LaMem train_1,
                              disjoint from the 6,000-image CLIP-only pool
  lamem_test2k_sample.json    test set: 2,000 images, uniformly at random from LaMem test_1,
                              excluding LaMem's eval2000 images

EDIT ME: point EMOSET_DIR at a copy of EmoSet with its train.json, and LAMEM_DIR at LaMem with its
splits/ folder. LAMEM_POOL_DIR is the folder holding the 6,000 CLIP-only train_1 images and
LAMEM_EVAL2000_DIR LaMem's eval2000 folder; both are only used to exclude their file names.
"""
import json, os, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 2026

# EDIT ME
EMOSET_DIR = "/path/to/EmoSet-118K"
LAMEM_DIR = "/path/to/LaMem"
LAMEM_POOL_DIR = "/path/to/LaMem/train6000"
LAMEM_EVAL2000_DIR = "/path/to/LaMem/eval2000"


def emoset_train2k():
    used = set(json.load(open(HERE / "features" / "emoset_extra_labels.json")))
    by = {}
    for lab, img, ann in json.load(open(f"{EMOSET_DIR}/train.json")):
        if Path(img).name not in used:
            by.setdefault(lab, []).append({"label": lab, "image_rel": img, "ann_rel": ann})
    rng, out = random.Random(SEED), []
    for lab in sorted(by):
        out += sorted(rng.sample(by[lab], 250), key=lambda d: d["image_rel"])
    return out


def lamem(split, exclude, n=2000):
    scores = dict(l.split() for l in open(f"{LAMEM_DIR}/splits/{split}_1.txt"))
    cands = sorted(k for k in scores if k not in exclude)
    random.Random(SEED).shuffle(cands)
    return [{"image": k, "score": float(scores[k])} for k in sorted(cands[:n])]


if __name__ == "__main__":
    out = {"emoset_train2k_sample.json": emoset_train2k(),
           "lamem_train2k_sample.json": lamem("train", set(os.listdir(LAMEM_POOL_DIR))),
           "lamem_test2k_sample.json": lamem("test", set(os.listdir(LAMEM_POOL_DIR)) | set(os.listdir(LAMEM_EVAL2000_DIR)))}
    for name, items in out.items():
        same = json.load(open(HERE / "samples" / name)) == items
        print(f"{name}: {len(items)} images, identical to the released list: {same}")
