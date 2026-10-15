"""
Prompt-robustness check (Gemma-3-4B, EmoSet): agreement between the original and the reworded prompts (prompts.py *_P2)
on the same images, per attribute (Spearman) and for the direct label (Cohen's kappa between prompts, and accuracy of each),
plus the main contrasts under each prompt from results/rev_gemma_linear_r30_ids_p2.json and ..._v_p2_ids_p2.json.
Run after:  IDS_VARIANT=_p2 python analysis.py gemma emoset   and   VARIANT=_p2 IDS_VARIANT=_p2 python analysis.py gemma emoset
Usage: python prompt_check.py  -> results/prompt_check_gemma.json
"""
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent; L = HERE / "labels"; R = HERE / "results"
EMO_NUM = ["valence_7", "arousal_5", "has_threat_or_danger", "has_disgusting_content", "novelty_5", "playfulness_5",
           "warmth_5", "energy_5", "aesthetic_5"]
PEOPLE = {"none": 0, "people_present": 1, "prominent_faces": 2}
rows = lambda f: {r["id"]: r for r in csv.DictReader(open(L / f))}
out = {"attributes": {}, "direct": {}, "contrasts": {}}
a = {**rows("gemma_emoset_features_labels.csv"), **rows("gemma_emoset2k_features_labels.csv")}
b = {**rows("gemma_emoset_features_p2_labels.csv"), **rows("gemma_emoset2k_features_p2_labels.csv")}
ids = sorted(set(a) & set(b))
out["n_images"] = len(ids)
for c in EMO_NUM:
    x = [float(a[i][c]) for i in ids]; y = [float(b[i][c]) for i in ids]
    out["attributes"][c] = {"spearman": round(float(spearmanr(x, y)[0]), 3), "exact_agreement": round(float(np.mean(np.array(x) == np.array(y))), 3)}
x = [PEOPLE[a[i]["has_people"]] for i in ids]; y = [PEOPLE[b[i]["has_people"]] for i in ids]
out["attributes"]["has_people"] = {"spearman": round(float(spearmanr(x, y)[0]), 3), "exact_agreement": round(float(np.mean(np.array(x) == np.array(y))), 3)}
da, db = rows("gemma_emoset_direct_labels.csv"), rows("gemma_emoset_direct_p2_labels.csv")
gt = {Path(s["image_rel"]).stem: s["label"] for s in json.load(open(HERE / "samples" / "emoset_sample.json"))}
di = sorted(set(da) & set(db) & set(gt))
out["direct"] = {"n": len(di),
                 "acc_original": round(float(np.mean([da[i]["emotion"] == gt[i] for i in di])), 4),
                 "acc_reworded": round(float(np.mean([db[i]["emotion"] == gt[i] for i in di])), 4),
                 "kappa_between_prompts": round(float(cohen_kappa_score([da[i]["emotion"] for i in di], [db[i]["emotion"] for i in di])), 3)}
for tag, f in (("original", "rev_gemma_linear_r30_ids_p2.json"), ("reworded", "rev_gemma_linear_r30_v_p2_ids_p2.json")):
    E = json.load(open(R / f))["emoset"]
    out["contrasts"][tag] = {"n_pool": E["n_pool"], "n_test": E["n_test"], "direct_acc": E["direct_acc"],
                             "curve": [{"n": r["n"], **{k: r["ci"][k][:4] for k in ("hybrid-pcak", "hybrid-pca2k", "attrs-pcak", "hybrid-clip")},
                                        "attrs": r["attrs"], "hybrid": r["hybrid"]} for r in E["curve"]]}
json.dump(out, open(R / "prompt_check_gemma.json", "w"), indent=1)
print("images", out["n_images"], "| attribute Spearman", {k: v["spearman"] for k, v in out["attributes"].items()})
print("direct", out["direct"])
for tag in ("original", "reworded"):
    print(tag, " ".join(f"{r['n']}:{r['hybrid-pcak'][0]:+.3f}" for r in out["contrasts"][tag]["curve"]))
