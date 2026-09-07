"""Merge the per-shard agentic-run JSONs back into one 400-question experiment file. The shards are
disjoint question sets (the dataset was split 8 x 50), so merging concatenates their `questions` arrays
and keeps one metadata block. Downstream scripts (overcite_ratio.py, r3_from_sidecar.py, agentic_loo.py)
enumerate questions positionally, so ordering by shard index is enough.

Usage: python merge_shards.py <out.json> <shard0.json> <shard1.json> ...
"""
import json, sys

OUT = sys.argv[1]
SHARDS = sys.argv[2:]
if not SHARDS:
    raise SystemExit("give at least one shard file")

merged = None
qs = []
for i, p in enumerate(SHARDS):
    d = json.load(open(p))
    if merged is None:
        merged = {k: v for k, v in d.items() if k != "questions"}
    qs.extend(d["questions"])
    print(f"  {p}: {len(d['questions'])} questions")

merged["questions"] = qs
json.dump(merged, open(OUT, "w"), indent=2)
print(f"wrote {OUT} with {len(qs)} questions from {len(SHARDS)} shards")
