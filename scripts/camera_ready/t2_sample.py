"""T2: sample ~100 (answer -> extracted entity set) pairs for a human agreement check.

Reviewer NbXB flagged that the entity extractor/canonicalizer is unvalidated and its noise
unquantified. This draws a stratified sample of answers paired with the entities the pipeline
extracted from them, so a human can judge extraction quality. If it lands, W7 (the Limitations
note about the extractor) shrinks to a sentence backed by a number.

Join: experiment_outputs (runs[].answer) x entity_extraction_output (runs[].extracted_entities)
on (question_id, iteration_number, run_id). The extractor is a fixed small model independent of
the generation model, so answer *diversity* (length, entity density) is what matters -- we
stratify across rounds (verbose early answers -> collapsed late answers) and two regimes.

Output: camera_ready_outputs/T2/t2_sample.jsonl -- one item per line:
  {item_id, source, question_id, question_text, round, run_id, answer, extracted_entities}
The answer text is shown to the annotator; extracted_entities is what they judge.
"""
import json, os, random

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
BASE = os.path.join(ROOT, "all_experiments", "graphite", "baseline")
OUT = os.path.join(ROOT, "camera_ready_outputs", "T2")
os.makedirs(OUT, exist_ok=True)
random.seed(7)

MODEL = "Qwen/Qwen2.5-14B-Instruct"
SOURCES = [
    ("replace_one", "local_replace_one"),
    ("search", "local_search"),
]
N_TARGET = 100

# round strata so we cover the whole collapse trajectory
def stratum(r, maxr):
    if r <= 1: return "early"
    if r >= maxr - 2: return "late"
    return "mid"


def load(regime, stem):
    exp = os.path.join(BASE, regime, "experiment_outputs", MODEL, f"{stem}.json")
    ent = os.path.join(BASE, regime, "entity_extraction_output", MODEL, f"{stem}_entity_results.json")
    if not (os.path.exists(exp) and os.path.exists(ent)):
        return []
    ed = json.load(open(exp)); nd = json.load(open(ent))
    # index extracted entities by (qid, iter, run_id)
    ext = {}
    for q in nd["questions"]:
        qid = q["question_id"]
        for it in q["iterations"]:
            for run in (it.get("runs") or []):
                ext[(qid, it["iteration_number"], run["run_id"])] = run.get("extracted_entities") or []
    maxr = max(it["iteration_number"] for it in ed["questions"][0]["iterations"])
    rows = []
    for q in ed["questions"]:
        qid = q["question_id"]; qt = q.get("question_text", "")
        for it in q["iterations"]:
            r = it["iteration_number"]
            if r == 0:
                continue  # round 0 is human-seeded; we assess loop answers
            for run in (it.get("runs") or []):
                key = (qid, r, run["run_id"])
                if key not in ext:
                    continue
                ans = (run.get("answer") or "").strip()
                if not ans:
                    continue
                rows.append(dict(source=f"{regime}", question_id=qid, question_text=qt,
                                 round=r, run_id=run["run_id"], answer=ans,
                                 extracted_entities=ext[key], _stratum=stratum(r, maxr)))
    return rows


pool = []
for regime, stem in SOURCES:
    pool.extend(load(regime, stem))
print(f"joined pool = {len(pool)} items")

# stratified sample: balance early/mid/late across the two regimes
buckets = {}
for row in pool:
    buckets.setdefault((row["source"], row["_stratum"]), []).append(row)
per_bucket = max(1, N_TARGET // max(1, len(buckets)))
sample = []
for key, rows in sorted(buckets.items()):
    random.shuffle(rows)
    sample.extend(rows[:per_bucket])
random.shuffle(sample)
sample = sample[:N_TARGET]

for i, row in enumerate(sample):
    row.pop("_stratum", None)
    row["item_id"] = f"t2_{i:03d}"

with open(os.path.join(OUT, "t2_sample.jsonl"), "w", encoding="utf-8") as f:
    for row in sample:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# quick coverage report
from collections import Counter
c = Counter((r["source"],) for r in sample)
print(f"sampled {len(sample)} items; by source:", dict(Counter(r['source'] for r in sample)))
print("round range:", min(r['round'] for r in sample), "..", max(r['round'] for r in sample))
print("avg extracted entities:", round(sum(len(r['extracted_entities']) for r in sample)/len(sample), 2))
print("wrote", os.path.join(OUT, "t2_sample.jsonl"))
