"""
Checks on the Qwen3.5-4B records behind Supplementary S6 and S7, written to results/qwen35_record_checks.json.
  test_run       the 18 images (3 per EmoSet and LaMem job) of the preliminary test run (checks/test_run/) against the
                 full run (labels/): identical records, the reasoning sentence included.
  relabel        checks/relabel_check.json, written by checks/relabel_check.py (reproduced as run): 21 EmoSet training-pool
                 images first labelled by the first, long-running process, relabelled in a fresh process.
  invalid_rating the one EmoSet direct rating counted invalid under the one-retry rule (checks/qwen35_excluded_rows.csv):
                 Qwen3.5-4B's EmoSet kappa and accuracy with it excluded (as reported), with the passing answer kept, and
                 with it scored as each wrong class, against Gemma-3-4B's.
Usage: python check_qwen35_records.py   (needs the EmoSet CLIP embeddings, as analysis.py does)
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent; CHK = HERE / "checks"
out = {}

# ---- preliminary test run against the full run ----
tr = {}
for f in sorted((CHK / "test_run").glob("qwen35_*_TEST_labels.csv")):
    job = f.name[len("qwen35_"):-len("_TEST_labels.csv")]
    full = {r["id"]: r for r in csv.DictReader(open(HERE / "labels" / f"qwen35_{job}_labels.csv"))}
    rows = list(csv.DictReader(open(f)))
    tr[job] = dict(n=len(rows), identical=sum(all(full.get(r["id"], {}).get(k) == v for k, v in r.items()) for r in rows))
out["test_run"] = dict(jobs=tr, n=sum(j["n"] for j in tr.values()), identical=sum(j["identical"] for j in tr.values()))

# ---- fresh-process relabelling ----
rc = json.load(open(CHK / "relabel_check.json"))
out["relabel"] = dict(job="emoset2k_features", rows=[r["row_index"] for r in rc], n=len(rc),
                      identical=sum(r["identical"] for r in rc))

# ---- the invalid EmoSet direct rating ----
ex = list(csv.DictReader(open(CHK / "qwen35_excluded_rows.csv")))
assert len(ex) == 1
sys.argv = ["x", "qwen35"]
import analysis as A
_, (ids, y, _, _, d), _, _ = A.emoset()
i = ids.index(ex[0]["id"]); assert d[i] == -1
ok = d >= 0
kap = lambda yy, dd: float(cohen_kappa_score(yy, dd))
acc = lambda yy, dd: float((yy == dd).mean())
inv = dict(id=ex[0]["id"], truth=A.EMO[y[i]], passing_answer=ex[0]["emotion"], n_test=len(ids),
           excluded=dict(n=int(ok.sum()), kappa=kap(y[ok], d[ok]), acc=acc(y[ok], d[ok])))
d2 = d.copy(); d2[i] = A.EMO.index(ex[0]["emotion"])
inv["passing_answer_kept"] = dict(n=len(ids), kappa=kap(y, d2), acc=acc(y, d2))
wrong = {}
for c in range(len(A.EMO)):
    if c != y[i]:
        d2 = d.copy(); d2[i] = c
        wrong[A.EMO[c]] = dict(kappa=kap(y, d2), acc=acc(y, d2))
inv["scored_wrong"] = dict(by_class=wrong, kappa_max=max(w["kappa"] for w in wrong.values()),
                           acc=wrong[next(iter(wrong))]["acc"])
zs = json.load(open(HERE / "tables" / "zero_shot.json"))
inv["gemma"] = dict(kappa=zs["emoset/gemma"]["agree"], acc=zs["emoset/gemma"]["acc"])
assert abs(inv["excluded"]["kappa"] - zs["emoset/qwen35"]["agree"]) < 1e-12, "tables/zero_shot.json is stale"
inv["qwen35_above_gemma"] = {k: inv[k]["kappa"] > inv["gemma"]["kappa"] for k in ("excluded", "passing_answer_kept")}
inv["qwen35_above_gemma"]["scored_wrong"] = inv["scored_wrong"]["kappa_max"] > inv["gemma"]["kappa"]
out["invalid_rating"] = inv
json.dump(out, open(HERE / "results" / "qwen35_record_checks.json", "w"), indent=1)
print(f"test run {out['test_run']['identical']}/{out['test_run']['n']} identical; relabel {out['relabel']['identical']}/{out['relabel']['n']} identical")
print(json.dumps({k: inv[k] for k in ("excluded", "passing_answer_kept", "gemma", "qwen35_above_gemma")}, indent=1),
      "scored wrong: kappa max", round(inv["scored_wrong"]["kappa_max"], 6), "acc", round(inv["scored_wrong"]["acc"], 6))
