"""R7 (entity-level): the decisive novelty-restriction test, on named entities instead of tokens.

Same logic as r7_novelty.py, but novel(D) is the set of ENTITIES the document introduced beyond its
source answer and the other context, and adoption is measured on the next answer's entities. Entities
are the unit the collapse claim is actually about, so this is the version that settles it.

Inputs:
  - answer entities: entity_extraction_output/.../local_replace_one_entity_results.json
    (questions[].iterations[].runs[].extracted_entities), already produced by the paper's extractor.
  - document entities: camera_ready_outputs/R7/r7_doc_entities.json  {qi: {doc_id: [entities]}},
    produced by ~/r7_extract_docs.py (job 63999619) with the SAME extractor. Fetch it here first.

Entities matched case-insensitively. Reports same-Q adoption, cross-Q placebo, and influence (Δ).
"""
import json, os, random
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
ANS = os.path.join(ROOT, "all_experiments", "graphite", "baseline", "replace_one",
                   "entity_extraction_output", "Qwen", "Qwen2.5-14B-Instruct", "local_replace_one_entity_results.json")
DOCENT = os.path.join(ROOT, "camera_ready_outputs", "R7", "r7_doc_entities.json")
random.seed(13)

def norm(es):
    return {str(e).strip().lower() for e in (es or []) if str(e).strip()}

if not os.path.exists(DOCENT):
    raise SystemExit(f"missing {DOCENT} -- fetch ~/r7_doc_entities.json from Unity (job 63999619) first")

doc_ent = json.load(open(DOCENT, encoding="utf-8"))                     # {qi: {doc_id: [ent]}}
ad = json.load(open(ANS, encoding="utf-8"))

# answer entities per question index: ans_runs[qi][round] = list of run entity-sets
ans_runs = []
for q in ad["questions"]:
    its = {}
    for it in q["iterations"]:
        its[it["iteration_number"]] = [norm(run.get("extracted_entities")) for run in (it.get("runs") or [])]
    ans_runs.append(its)
n_q = len(ans_runs)

def novel_entities(qi, k):
    de = doc_ent.get(str(qi), {})
    if f"gen_{k}_0" not in de:
        return None
    dset = norm(de[f"gen_{k}_0"])
    if not dset:
        return None
    src = set().union(*ans_runs[qi].get(k - 1, [set()])) if ans_runs[qi].get(k - 1) else set()
    other = set()
    for did, es in de.items():
        if did != f"gen_{k}_0":
            other |= norm(es)
    return dset - src - other

def adoption(novel, runs):
    if not novel or not runs:
        return None
    return sum(len(novel & r) / len(novel) for r in runs) / len(runs)

by_round = defaultdict(lambda: {"same": [], "cross": [], "nsize": []})
for qi in range(n_q):
    for k in range(1, 20):
        novel = novel_entities(qi, k)
        if not novel:
            continue
        same = adoption(novel, ans_runs[qi].get(k))
        if same is None:
            continue
        qj = qi
        for _ in range(8):
            qj = random.randrange(n_q)
            if qj != qi and ans_runs[qj].get(k):
                break
        cross = adoption(novel, ans_runs[qj].get(k))
        by_round[k]["same"].append(same)
        if cross is not None:
            by_round[k]["cross"].append(cross)
        by_round[k]["nsize"].append(len(novel))

def m(v):
    return sum(v) / len(v) if v else float("nan")

print(f"{'round':>5} {'n':>5} {'novelEnt':>8} {'same-Q':>8} {'cross-Q':>8} {'influence':>10}")
alls, allc = [], []
for k in sorted(by_round):
    b = by_round[k]
    s = m(b["same"]); c = m(b["cross"])
    alls += b["same"]; allc += b["cross"]
    print(f"{k:5d} {len(b['same']):5d} {m(b['nsize']):8.2f} {s:8.3f} {c:8.3f} {s-c:+10.3f}")
print("-" * 50)
print(f"{'POOL':>5} {len(alls):5d} {'':8} {m(alls):8.3f} {m(allc):8.3f} {m(alls)-m(allc):+10.3f}")
print("\nsame-Q = fraction of D's invented ENTITIES the next answer adopts (mean over runs).")
print("influence = same - cross: entity adoption that requires D in context, not ancestry.")
