"""T1 step 1 (backend-independent): sample stored (question, round, context) triples for
re-prompting with direct citation elicitation, and carry each triple's LOO citations along so
the two attribution methods can be compared on the SAME items.

Source: the reported Qwen-14B Replace-One run (LOO citations already stored per run). For each
sampled (question, round) we record the numbered document list (with provenance) and the LOO
per-run cited doc_ids. t1_generate.py will re-prompt the model over these contexts; t1_compare.py
puts the direct-elicited citations next to the LOO ones (over-citation ratio + doc-level Jaccard).

Output: camera_ready_outputs/T1/t1_sample.jsonl
  {item_id, question_id, question_text, round, docs:[{n,doc_id,self_gen,text}], loo_runs:[[doc_id,...], ...]}
`n` is the 1-based number shown to the model in the prompt; the model cites by `n`.
"""
import json, os, random

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SRC = os.path.join(ROOT, "all_experiments", "graphite", "baseline", "replace_one",
                   "experiment_outputs", "Qwen", "Qwen2.5-14B-Instruct", "local_replace_one.json")
OUT = os.path.join(ROOT, "camera_ready_outputs", "T1")
os.makedirs(OUT, exist_ok=True)
random.seed(11)
N_TARGET = 300
DOC_CHARS = 400  # match the run's chars-per-doc so the re-prompt sees the same context


def is_ai(doc_id, url, it):
    doc_id = str(doc_id or "").lower(); url = str(url or "").lower()
    return doc_id.startswith("gen_") or url == "model_generated" or (isinstance(it, int) and it > 0 and not doc_id.startswith("ref_"))


d = json.load(open(SRC))
maxr = max(it["iteration_number"] for it in d["questions"][0]["iterations"])

pool = []
for q in d["questions"]:
    qid = q["question_id"]; qt = q.get("question_text", "")
    for it in q["iterations"]:
        r = it["iteration_number"]
        if r == 0:
            continue
        docs = it.get("documents") or []
        runs = it.get("runs") or []
        if not docs or not runs:
            continue
        doclist = [dict(n=i + 1, doc_id=dc.get("doc_id"),
                        self_gen=bool(is_ai(dc.get("doc_id"), dc.get("url"), dc.get("iteration"))),
                        text=(dc.get("text") or "")[:DOC_CHARS])
                   for i, dc in enumerate(docs)]
        loo_runs = [[c.get("doc_id") for c in (run.get("citations") or []) if c.get("doc_id")]
                    for run in runs]
        pool.append(dict(question_id=qid, question_text=qt, round=r, docs=doclist, loo_runs=loo_runs,
                         _stratum=("early" if r <= 1 else "late" if r >= maxr - 2 else "mid")))

# stratify across rounds so early (mixed context) and late (heavily self-gen) are both covered
buckets = {}
for row in pool:
    buckets.setdefault(row["_stratum"], []).append(row)
per = max(1, N_TARGET // len(buckets))
sample = []
for k, rows in buckets.items():
    random.shuffle(rows); sample.extend(rows[:per])
random.shuffle(sample); sample = sample[:N_TARGET]
for i, row in enumerate(sample):
    row.pop("_stratum", None); row["item_id"] = f"t1_{i:03d}"

with open(os.path.join(OUT, "t1_sample.jsonl"), "w", encoding="utf-8") as f:
    for row in sample:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

sg = sum(sum(1 for dc in r["docs"] if dc["self_gen"]) for r in sample)
tot = sum(len(r["docs"]) for r in sample)
print(f"sampled {len(sample)} (question,round) triples")
print(f"context self-gen fraction across sample = {sg}/{tot} = {sg/tot:.3f}")
print(f"round range {min(r['round'] for r in sample)}..{max(r['round'] for r in sample)}")
print("wrote", os.path.join(OUT, "t1_sample.jsonl"))
