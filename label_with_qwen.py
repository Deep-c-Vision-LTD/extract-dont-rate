"""
Labels from the open-weights VLM (Qwen2.5-VL-7B-Instruct, Apache-2.0), run locally (Apple-silicon
MPS, CUDA, or CPU). Same jobs, prompts, and schemas as the Claude run (label_with_vlm.py /
prompts.py); tool calling is replaced by a JSON answer validated against the same schema. Greedy
decoding, so labels are deterministic; an invalid answer gets one sampled retry. Resumable: rows
are appended to labels/qwen_<job>_labels.csv and finished ids are skipped on restart.

Free but slow: about 5 s per direct rating and 11 to 14 s per attribute extraction on an M4 Pro.
All outputs are already included in labels/.

Usage:  python label_with_qwen.py <job> [<job> ...]     (or "all"; "test" = 3 images per job)
"""
import csv, json, os, re, sys, time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from label_with_vlm import JOBS

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
ORDER = list(JOBS)


def schema_text(tool):
    props, lines = tool["input_schema"]["properties"], []
    for k, v in props.items():
        if v["type"] == "integer":
            rng = f"integer {v['minimum']}-{v['maximum']}"
        elif "enum" in v:
            rng = "one of " + ", ".join(f'"{e}"' for e in v["enum"])
        else:
            rng = "string"
        lines.append(f'  "{k}": {rng}. {v.get("description", "")}')
    return "{\n" + "\n".join(lines) + "\n}"


def build_prompt(job):
    tool = job["tool"]
    text = re.sub(r"then call \w+", "then answer", job["prompt"])
    text = re.sub(r",? ?then call \w+\.?", ".", text)
    return (f"{text}\n\nAnswer with ONLY a JSON object with exactly these keys, in this order "
            f"(\"reasoning\" is one short sentence):\n{schema_text(tool)}")


def validate(obj, tool):
    props = tool["input_schema"]["properties"]
    row = {}
    for k, v in props.items():
        if k not in obj:
            return None
        x = obj[k]
        if v["type"] == "integer":
            try:
                x = int(round(float(x)))
            except (TypeError, ValueError):
                return None
            if not v["minimum"] <= x <= v["maximum"]:
                return None
        elif "enum" in v and x not in v["enum"]:
            return None
        row[k] = x
    return row


def parse(text, tool):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return validate(json.loads(m.group(0)), tool)
    except json.JSONDecodeError:
        return None


def load_image(path, max_edge=768):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if max(w, h) > max_edge:
        s = max_edge / max(w, h)
        im = im.resize((int(w * s), int(h * s)), Image.BILINEAR)
    return im


def run(names, limit=None):
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28)
    for name in names:
        job = JOBS[name]
        prompt = build_prompt(job)
        out = HERE / "labels" / f"qwen_{name}_labels.csv"
        keys = ["id"] + list(job["tool"]["input_schema"]["properties"].keys())
        done = set()
        if out.exists():
            done = {r["id"] for r in csv.DictReader(open(out))}
        items = json.load(open(HERE / "samples" / job["sample"]))
        if limit:
            items = items[:limit]
        todo = [it for it in items
                if os.path.splitext(os.path.basename(it[job["img_key"]]))[0] not in done]
        print(f"[{name}] {len(done)} done, {len(todo)} to go", flush=True)
        new_file = not out.exists()
        fh = open(out, "a", newline="")
        wr = csv.DictWriter(fh, fieldnames=keys)
        if new_file:
            wr.writeheader()
        t0, fails = time.time(), 0
        for i, it in enumerate(todo):
            cid = os.path.splitext(os.path.basename(it[job["img_key"]]))[0]
            img = load_image(os.path.join(job["img_base"], it[job["img_key"]]))
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            chat = proc.apply_chat_template(msgs, add_generation_prompt=True)
            row = None
            for attempt in range(2):  # one retry with sampling if the greedy answer is invalid
                inp = proc(text=[chat], images=[img], return_tensors="pt").to(dev)
                with torch.no_grad():
                    gen = model.generate(**inp, max_new_tokens=220, do_sample=attempt > 0,
                                         temperature=0.7 if attempt else None)
                txt = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                row = parse(txt, job["tool"])
                if row:
                    break
            if row is None:
                fails += 1
                continue
            wr.writerow({"id": cid, **row})
            fh.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                rate = (time.time() - t0) / (i + 1)
                print(f"  [{name}] {i+1}/{len(todo)}  {rate:.1f}s/img  eta {rate*(len(todo)-i-1)/60:.0f} min"
                      f"  invalid {fails}", flush=True)
        fh.close()
        print(f"[{name}] finished, invalid after retry: {fails}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:] or ["test"]
    if args == ["test"]:
        run(ORDER, limit=3)
    else:
        run(ORDER if args == ["all"] else args)
