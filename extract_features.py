"""
Extract frozen CLIP ViT-L/14 image embeddings (768-d, L2-normalized, no text tower) for every image
set in the study. The LaMem embeddings are included in features/. EmoSet's terms do not allow redistributing derived
data, so the EmoSet embeddings and the EmoSet pool's label file are NOT included: run "python extract_features.py emoset"
on your copy of EmoSet to create them (the image lists in samples/ fix which images and in which order).

  features/emoset_clip.npz           EmoSet test set (1,600)
  features/emoset_train2k_clip.npz   EmoSet training pool (2,000)
  features/emoset_extra_clip.npz     EmoSet CLIP-only pool (20,000; unlabelled for PCA, labels for the large-budget reference)
  features/emoset_extra_labels.json  labels of that pool (from the image folder names), used only for the large-budget reference
  features/lamem_train2k_clip.npz    LaMem training pool (2,000)
  features/lamem_test2k_clip.npz     LaMem test set (2,000)
  features/lamem_train6000_clip.npz  LaMem CLIP-only pool (6,000)

EDIT ME: set EMOSET_IMAGE_DIR / LAMEM_IMAGE_DIR to your copies of the raw benchmarks. Neither
dataset's images are redistributed here:
  EmoSet: https://vcc.tech/EmoSet
  LaMem:  http://memorability.csail.mit.edu/
"""
import json, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPVisionModelWithProjection

HERE = Path(__file__).resolve().parent

# EDIT ME
EMOSET_IMAGE_DIR = "/path/to/EmoSet-118K"          # expects <dir>/image/<label>/<file>.jpg layout
LAMEM_IMAGE_DIR = "/path/to/LaMem/images"           # all LaMem images, <dir>/<file>.jpg

M = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
S = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval()


def prep(p):
    im = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
    return ((np.asarray(im, np.float32) / 255 - M) / S).transpose(2, 0, 1)


def extract(paths, out_npz):
    names, embs = [], []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            batch = paths[i:i + 32]
            x = torch.from_numpy(np.stack([prep(p) for p in batch]))
            e = clip(pixel_values=x).image_embeds.numpy()
            e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
            names += [os.path.basename(p) for p in batch]
            embs.append(e)
            if (i // 32) % 5 == 0:
                print(f"  {i + len(batch)}/{len(paths)}", flush=True)
    np.savez(out_npz, names=np.array(names), emb=np.vstack(embs))
    print(f"saved {out_npz}: {len(names)} embeddings")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["emoset", "lamem"]
    F, S_ = HERE / "features", HERE / "samples"
    if "emoset" in which:
        for sample, out in (("emoset_sample.json", "emoset_clip.npz"), ("emoset_train2k_sample.json", "emoset_train2k_clip.npz")):
            items = json.load(open(S_ / sample))
            extract([f"{EMOSET_IMAGE_DIR}/{e['image_rel']}" for e in items], F / out)
        extra = [e["image_rel"] for e in json.load(open(S_ / "emoset_extra_pool.json"))]   # image/<label>/<file>.jpg
        json.dump({os.path.basename(p): p.split("/")[1] for p in extra}, open(F / "emoset_extra_labels.json", "w"))
        extract([f"{EMOSET_IMAGE_DIR}/{p}" for p in extra], F / "emoset_extra_clip.npz")
    if "lamem" in which:
        for sample, out in (("lamem_train2k_sample.json", "lamem_train2k_clip.npz"), ("lamem_test2k_sample.json", "lamem_test2k_clip.npz")):
            items = json.load(open(S_ / sample))
            extract([f"{LAMEM_IMAGE_DIR}/{e['image']}" for e in items], F / out)
        pool = json.load(open(F / "lamem_train6000_scores.json"))           # file name -> score
        extract([f"{LAMEM_IMAGE_DIR}/{name}" for name in sorted(pool)], F / "lamem_train6000_clip.npz")
