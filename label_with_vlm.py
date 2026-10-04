"""
Label the EmoSet and LaMem images with Claude Sonnet 4.6 via the Anthropic Batch API (50%
cheaper than real-time). Six label sets, one batch each:
  emoset_features    : emotion attributes, EmoSet test set (validation subset)  1,600 images
  emoset_direct      : direct 8-way emotion rating, EmoSet test set              1,600 images
  emoset2k_features  : emotion attributes, EmoSet training pool                  2,000 images
  lamemtr2k_features : memorability attributes, LaMem training pool              2,000 images
  lamem2k_features   : memorability attributes, LaMem test set                   2,000 images
  lamem2k_direct     : direct 1-10 memorability rating, LaMem test set           2,000 images

Part of "Full Reproduction" only: this is the step that costs money and needs an API key.
Every CSV it would write is already included in labels/, so the Quick Reproduction never
calls it. The open-weights labels (labels/qwen_*) come from label_with_qwen.py.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python label_with_vlm.py submit <name|all>
  python label_with_vlm.py fetch  <name|all>

Cost: about 11,200 batched calls, roughly $40 at the time of writing.

EDIT ME: set EMOSET_DIR / LAMEM_DIR to your local copies of the raw datasets.
"""
import base64, csv, io, json, os, sys, time
from pathlib import Path
from PIL import Image
from prompts import (DIRECT_EMOTION_TOOL, DIRECT_EMOTION_PROMPT,
                      EMOTION_FEATURE_TOOL, EMOTION_FEATURE_PROMPT,
                      DIRECT_MEMORABILITY_TOOL, DIRECT_MEMORABILITY_PROMPT,
                      MEMORY_FEATURE_TOOL, MEMORY_FEATURE_PROMPT)

HERE = Path(__file__).resolve().parent
MODEL = "claude-sonnet-4-6"

# EDIT ME
EMOSET_DIR = "/path/to/EmoSet-118K"     # expects <dir>/image/<label>/<file>.jpg
LAMEM_DIR = "/path/to/LaMem/images"     # all LaMem images, <dir>/<file>.jpg


def _job(sample, key, base, tool, prompt, name):
    return dict(sample=sample, img_key=key, img_base=base, tool=tool, prompt=prompt, tool_name=name)


JOBS = {
    "emoset_features":    _job("emoset_sample.json", "image_rel", EMOSET_DIR, EMOTION_FEATURE_TOOL, EMOTION_FEATURE_PROMPT, "record_emotion_features"),
    "emoset_direct":      _job("emoset_sample.json", "image_rel", EMOSET_DIR, DIRECT_EMOTION_TOOL, DIRECT_EMOTION_PROMPT, "classify_emotion"),
    "emoset2k_features":  _job("emoset_train2k_sample.json", "image_rel", EMOSET_DIR, EMOTION_FEATURE_TOOL, EMOTION_FEATURE_PROMPT, "record_emotion_features"),
    "lamemtr2k_features": _job("lamem_train2k_sample.json", "image", LAMEM_DIR, MEMORY_FEATURE_TOOL, MEMORY_FEATURE_PROMPT, "record_memory_features"),
    "lamem2k_features":   _job("lamem_test2k_sample.json", "image", LAMEM_DIR, MEMORY_FEATURE_TOOL, MEMORY_FEATURE_PROMPT, "record_memory_features"),
    "lamem2k_direct":     _job("lamem_test2k_sample.json", "image", LAMEM_DIR, DIRECT_MEMORABILITY_TOOL, DIRECT_MEMORABILITY_PROMPT, "rate_memorability"),
}


def encode(path, max_edge=768):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if max(w, h) > max_edge:
        s = max_edge / max(w, h)
        im = im.resize((int(w * s), int(h * s)), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def submit(name):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    job = JOBS[name]
    client = anthropic.Anthropic()
    items = json.load(open(HERE / "samples" / job["sample"]))
    print(f"[{name}] building {len(items)} requests...")
    reqs = []
    for it in items:
        img_path = os.path.join(job["img_base"], it[job["img_key"]])
        cid = os.path.splitext(os.path.basename(it[job["img_key"]]))[0]
        reqs.append(Request(
            custom_id=cid,
            params=MessageCreateParamsNonStreaming(
                model=MODEL, max_tokens=300, tools=[job["tool"]],
                tool_choice={"type": "tool", "name": job["tool_name"]},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encode(img_path)}},
                    {"type": "text", "text": job["prompt"]}]}])))
    batch = client.messages.batches.create(requests=reqs)
    (HERE / f"batch_id_{name}.txt").write_text(batch.id)
    print(f"[{name}] submitted batch {batch.id} (status {batch.processing_status})")


def fetch(name):
    import anthropic
    client = anthropic.Anthropic()
    bid = (HERE / f"batch_id_{name}.txt").read_text().strip()
    while True:
        b = client.messages.batches.retrieve(bid)
        if b.processing_status == "ended":
            break
        print(f"  [{name}] {b.processing_status}: {b.request_counts.processing} processing, "
              f"{b.request_counts.succeeded} done", flush=True)
        time.sleep(60)
    rows = []
    for res in client.messages.batches.results(bid):
        if res.result.type != "succeeded":
            continue
        tool_use = next((c for c in res.result.message.content if c.type == "tool_use"), None)
        if tool_use is None:
            continue
        row = {"id": res.custom_id, **tool_use.input}
        rows.append(row)
    out = HERE / "labels" / f"{name}_labels.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    c = client.messages.batches.retrieve(bid)
    print(f"[{name}] wrote {len(rows)} labels -> {out}  (succeeded {c.request_counts.succeeded}, errored {c.request_counts.errored})")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("submit", "fetch"):
        sys.exit("usage: python label_with_vlm.py submit|fetch <" + "|".join(JOBS) + "|all>")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("set ANTHROPIC_API_KEY")
    action, name = sys.argv[1], sys.argv[2]
    names = list(JOBS.keys()) if name == "all" else [name]
    for n in names:
        (submit if action == "submit" else fetch)(n)
