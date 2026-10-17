"""Re-label images already labelled by the full Qwen3.5 run (same code path as label_with_open_vlm.run, qwen35 branch,
with the MPS cache release) and compare with the stored rows: does CPU contention, swapping or the restart change outputs?"""
import csv, json, os, sys
import torch
sys.path.insert(0, "/Users/hamitsoyel/Downloads/xsoyel_perceptual/paper_v2/extract-dont-rate")
import label_with_open_vlm as L
from transformers import AutoModelForImageTextToText, AutoProcessor
job = "emoset2k_features"; J = L.JOBS[job]
rows = list(csv.DictReader(open(L.REPO / "labels" / f"qwen35_{job}_labels.csv")))
# rows 200-399: CPU-contention period (~17-20 s/img); rows 950-1046: heaviest swapping, just before the restart
pick = rows[200:400:25] + rows[950:1047:8]
items = {os.path.splitext(os.path.basename(it[J["img_key"]]))[0]: it for it in json.load(open(L.REPO / "samples" / J["sample"]))}
dev = "mps"; mid = L.MODELS["qwen35"]
model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16).to(dev).eval()
proc = AutoProcessor.from_pretrained(mid)
prompt = L.build_prompt(J)
out = []
for k, r in enumerate(pick):
    if k:
        torch.mps.empty_cache()
    img = L.load_image(os.path.join(L.BASE[job], items[r["id"]][J["img_key"]]))
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                                   enable_thinking=False).to(dev)
    inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=220, do_sample=False, temperature=None)
    row = L.parse(proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0], J["tool"])
    same = row is not None and all(str(row[c]) == r[c] for c in row)
    attrs = row is not None and all(str(row[c]) == r[c] for c in row if c != "reasoning")
    out.append(dict(id=r["id"], row_index=rows.index(r), identical=same, attributes_identical=attrs,
                    diff={c: (r[c], row[c]) for c in row if str(row[c]) != r[c]} if row else "INVALID"))
    print(out[-1]["row_index"], r["id"], "identical" if same else ("attributes identical" if attrs else f"DIFF {out[-1]['diff']}"), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "relabel_check.json"), "w"), indent=1)
print(f"identical {sum(o['identical'] for o in out)}/{len(out)}; attributes identical {sum(o['attributes_identical'] for o in out)}/{len(out)}")
