"""
Labels from a further open-weights VLM, run locally, with the identical jobs, prompts, schemas, JSON validation
and decoding policy as label_with_qwen.py (greedy; one sampled retry if the answer is invalid; resumable).
Writes extract-dont-rate/labels/<prefix>_<job>_labels.csv.

Usage: python label_with_open_vlm.py gemma <job> [<job> ...]   ("all" = original jobs, "new" = new-dataset jobs, "p2" = reworded emotion prompts on EmoSet, "test" = 3 images per job)
Models: gemma = google/gemma-3-4b-it (Gemma terms of use; gated on Hugging Face);
        qwen = Qwen/Qwen2.5-VL-7B-Instruct with label_with_qwen.py's processor settings (for the new datasets);
        qwen35 = Qwen/Qwen3.5-4B (Apache-2.0) with thinking mode disabled (enable_thinking=False), see PREREGISTRATION_qwen35.md.
New-dataset jobs (CVIU resubmission): artphoto_/abstract_/memcat_features on pool + test, *_direct on test only,
with the unchanged EmoSet emotion and LaMem memorability prompts and schemas.
"""
import csv, json, os, sys, time
import gc
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from label_with_vlm import JOBS, _job
from prompts import (DIRECT_EMOTION_TOOL, DIRECT_EMOTION_PROMPT, EMOTION_FEATURE_TOOL, EMOTION_FEATURE_PROMPT,
                     DIRECT_MEMORABILITY_TOOL, DIRECT_MEMORABILITY_PROMPT, MEMORY_FEATURE_TOOL, MEMORY_FEATURE_PROMPT,
                     EMOTION_FEATURE_TOOL_P2, EMOTION_FEATURE_PROMPT_P2, DIRECT_EMOTION_TOOL_P2, DIRECT_EMOTION_PROMPT_P2)
from label_with_qwen import build_prompt, parse, load_image

MODELS = {"gemma": "google/gemma-3-4b-it", "qwen": "Qwen/Qwen2.5-VL-7B-Instruct", "qwen35": "Qwen/Qwen3.5-4B"}
# EDIT ME: image folders (EmoSet: <dir>/image/<label>/<file>.jpg; LaMem: all images in one folder)
EMOSET_DIR = os.environ.get("EMOSET_DIR", "/path/to/EmoSet-118K")
LAMEM_TEST_DIR = os.environ.get("LAMEM_DIR", "/path/to/LaMem/images")
BASE = {"emoset_features": EMOSET_DIR, "emoset_direct": EMOSET_DIR, "emoset2k_features": EMOSET_DIR,
        "lamemtr2k_features": LAMEM_TEST_DIR, "lamem2k_features": LAMEM_TEST_DIR, "lamem2k_direct": LAMEM_TEST_DIR}
NEW = Path(os.environ.get("NEW_DATA", "/path/to/cviu_new_datasets"))     # EDIT ME, as in prep_new_datasets.py
ART, ABS, MEM = str(NEW / "artphoto" / "testImages_artphoto"), str(NEW / "abstract" / "testImages_abstract"), str(NEW / "memcat" / "MemCat")
JOBS.update({
    "artphoto_features": _job("artphoto_all_sample.json", "image", ART, EMOTION_FEATURE_TOOL, EMOTION_FEATURE_PROMPT, "record_emotion_features"),
    "artphoto_direct": _job("artphoto_test_sample.json", "image", ART, DIRECT_EMOTION_TOOL, DIRECT_EMOTION_PROMPT, "classify_emotion"),
    "abstract_features": _job("abstract_all_sample.json", "image", ABS, EMOTION_FEATURE_TOOL, EMOTION_FEATURE_PROMPT, "record_emotion_features"),
    "abstract_direct": _job("abstract_test_sample.json", "image", ABS, DIRECT_EMOTION_TOOL, DIRECT_EMOTION_PROMPT, "classify_emotion"),
    "memcat_features": _job("memcat_all_sample.json", "image", MEM, MEMORY_FEATURE_TOOL, MEMORY_FEATURE_PROMPT, "record_memory_features"),
    "memcat_direct": _job("memcat_test_sample.json", "image", MEM, DIRECT_MEMORABILITY_TOOL, DIRECT_MEMORABILITY_PROMPT, "rate_memorability"),
})
BASE.update({"artphoto_features": ART, "artphoto_direct": ART, "abstract_features": ABS, "abstract_direct": ABS,
             "memcat_features": MEM, "memcat_direct": MEM})
# prompt-robustness check: the reworded emotion prompts on EmoSet (same images as the original jobs)
JOBS.update({
    "emoset_features_p2": _job("emoset_sample.json", "image_rel", EMOSET_DIR, EMOTION_FEATURE_TOOL_P2, EMOTION_FEATURE_PROMPT_P2, "record_emotion_features"),
    "emoset2k_features_p2": _job("emoset_train2k_sample.json", "image_rel", EMOSET_DIR, EMOTION_FEATURE_TOOL_P2, EMOTION_FEATURE_PROMPT_P2, "record_emotion_features"),
    "emoset_direct_p2": _job("emoset_sample.json", "image_rel", EMOSET_DIR, DIRECT_EMOTION_TOOL_P2, DIRECT_EMOTION_PROMPT_P2, "classify_emotion"),
})
BASE.update({"emoset_features_p2": EMOSET_DIR, "emoset2k_features_p2": EMOSET_DIR, "emoset_direct_p2": EMOSET_DIR})
P2_ORDER = ["emoset_features_p2", "emoset2k_features_p2", "emoset_direct_p2"]
NEW_ORDER = ["artphoto_features", "abstract_features", "artphoto_direct", "abstract_direct", "memcat_features", "memcat_direct"]
ORDER = ["emoset_features", "emoset2k_features", "emoset_direct", "lamem2k_features", "lamemtr2k_features", "lamem2k_direct"]


MAX_PER_RUN = int(os.environ.get("MAX_PER_RUN", 0))       # 0 = never recycle
DONE_THIS_RUN = 0


def run(prefix, names, limit=None):
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    mid = MODELS[prefix]
    model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16).to(dev).eval()
    proc = (AutoProcessor.from_pretrained(mid, min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28) if prefix == "qwen"
            else AutoProcessor.from_pretrained(mid))
    for name in names:
        job = JOBS[name]
        prompt = build_prompt(job)
        out = REPO / "labels" / f"{prefix}_{name}{'_TEST' if limit else ''}_labels.csv"
        keys = ["id"] + list(job["tool"]["input_schema"]["properties"].keys())
        done = {r["id"] for r in csv.DictReader(open(out))} if out.exists() else set()
        items = json.load(open(REPO / "samples" / job["sample"]))[: limit or None]
        todo = [it for it in items if os.path.splitext(os.path.basename(it[job["img_key"]]))[0] not in done]
        print(f"[{prefix} {name}] {len(done)} done, {len(todo)} to go", flush=True)
        new_file = not out.exists()
        fh = open(out, "a", newline=""); wr = csv.DictWriter(fh, fieldnames=keys)
        if new_file:
            wr.writeheader()
        t0, fails = time.time(), 0
        for i, it in enumerate(todo):
            if dev == "mps" and i:                # free the MPS buffer cache: inputs vary in size, so cached buffers pile up
                torch.mps.empty_cache()           # (memory only; outputs are unchanged)
                if i % 25 == 0:
                    gc.collect()
            cid = os.path.splitext(os.path.basename(it[job["img_key"]]))[0]
            img = load_image(os.path.join(BASE[name], it[job["img_key"]]))
            if prefix == "qwen":                  # exactly as label_with_qwen.py
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                chat = proc.apply_chat_template(msgs, add_generation_prompt=True)
                inp = proc(text=[chat], images=[img], return_tensors="pt").to(dev)
            else:
                msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
                extra = {"enable_thinking": False} if prefix == "qwen35" else {}
                inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", **extra).to(dev)
            if "pixel_values" in inp:
                inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
            row = None
            for attempt in range(2):
                with torch.no_grad():
                    gen = model.generate(**inp, max_new_tokens=220, do_sample=attempt > 0,
                                         temperature=0.7 if attempt else None)
                txt = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                row = parse(txt, job["tool"])
                if row:
                    break
            if row is None:
                fails += 1
                if limit:
                    print("  INVALID:", txt[:300].replace("\n", " "), flush=True)
                continue
            wr.writerow({"id": cid, **row}); fh.flush()
            global DONE_THIS_RUN
            DONE_THIS_RUN += 1
            if MAX_PER_RUN and DONE_THIS_RUN >= MAX_PER_RUN:     # memory grows per new input shape on MPS: exit and let
                fh.close()                                     # the wrapper restart (resume skips done ids; outputs
                print(f"[{prefix} {name}] recycling process after {DONE_THIS_RUN} images", flush=True)
                sys.exit(75)                                   # are unchanged across restarts)
            if limit or (i + 1) % 25 == 0 or i + 1 == len(todo):
                rate = (time.time() - t0) / (i + 1)
                print(f"  [{prefix} {name}] {i+1}/{len(todo)}  {rate:.1f}s/img  eta {rate*(len(todo)-i-1)/60:.0f} min"
                      f"  invalid {fails}", flush=True)
        fh.close()
        print(f"[{prefix} {name}] finished, invalid after retry: {fails}", flush=True)


if __name__ == "__main__":
    prefix, args = sys.argv[1], sys.argv[2:] or ["test"]
    if args == ["test"]:
        run(prefix, ORDER, limit=3)
    else:
        run(prefix, ORDER if args == ["all"] else NEW_ORDER if args == ["new"] else P2_ORDER if args == ["p2"] else args)
