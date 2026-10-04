"""Encoder robustness: embed every image set with two other frozen encoders (DINOv2-L, self-supervised
and vision-only; SigLIP-L, image-text) so the analysis can be rerun with each in place of CLIP.
Writes features/<set>_<enc>.npz (not included in the repository because of their size, about 140 MB
per encoder; roughly 30 minutes per encoder on an Apple-silicon GPU).
Usage: python extract_alt_encoders.py [dinov2l] [siglipl]
EDIT ME: EM / LM point at the raw EmoSet and LaMem images."""
import json, sys, numpy as np, torch
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
HERE = Path(__file__).resolve().parent
# EDIT ME
EM = "/path/to/EmoSet-118K"
LM = "/path/to/LaMem/images"
ENCS = {"dinov2l": "facebook/dinov2-large", "siglipl": "google/siglip-large-patch16-256"}
dev = "mps" if torch.backends.mps.is_available() else "cpu"

def sets():
    out = []
    for s, o in (("emoset_sample.json", "emoset"), ("emoset_train2k_sample.json", "emoset_train2k")):
        out.append((o, [f"{EM}/{e['image_rel']}" for e in json.load(open(HERE / "samples" / s))]))
    ex = np.load(HERE / "features" / "emoset_extra_clip.npz")["names"]; lab = json.load(open(HERE / "features" / "emoset_extra_labels.json"))
    out.append(("emoset_extra", [f"{EM}/image/{lab[str(n)]}/{n}" for n in ex]))
    for s, o in (("lamem_train2k_sample.json", "lamem_train2k"), ("lamem_test2k_sample.json", "lamem_test2k")):
        out.append((o, [f"{LM}/{e['image']}" for e in json.load(open(HERE / "samples" / s))]))
    pc = np.load(HERE / "features" / "lamem_train6000_clip.npz")["names"]
    out.append(("lamem_train6000", [f"{LM}/{n}" for n in pc]))
    return out

for enc, hf in ENCS.items():
    if len(sys.argv) > 1 and enc not in sys.argv[1:]:
        continue
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base" if enc == "dinov2l" else hf, local_files_only=True)  # DINOv2-L shares DINOv2-B preprocessing
    m = AutoModel.from_pretrained(hf, local_files_only=True).to(dev).eval()
    for name, paths in sets():
        embs = []
        with torch.no_grad():
            for i in range(0, len(paths), 64):
                ims = [Image.open(p).convert("RGB") for p in paths[i:i + 64]]
                x = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
                if enc.startswith("siglip"):
                    e = m.get_image_features(pixel_values=x)
                    e = e if torch.is_tensor(e) else e.pooler_output   # newer transformers return an output object
                else:
                    e = m(pixel_values=x).pooler_output
                e = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy(); embs.append(e)
        names = np.array([p.rsplit("/", 1)[1] for p in paths])
        np.savez(HERE / "features" / f"{name}_{enc}.npz", names=names, emb=np.vstack(embs))
        print(enc, name, len(names), np.vstack(embs).shape[1], flush=True)
print("done")
