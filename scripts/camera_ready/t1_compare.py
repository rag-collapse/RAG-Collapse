"""T1 step 3: put the two attribution methods side by side on the SAME items.

Method A = LOO (post-hoc, already stored per run in t1_sample.jsonl -> loo_runs).
Method B = direct elicitation (t1_direct.jsonl, produced by t1_generate.py).

Reports, over the sampled (question, round) triples:
  - over-citation ratio under each method = (self-gen share of citations) / (self-gen share of context)
  - the two ratios side by side  -> "one effect measured two ways" as an in-paper statistic (C2)
  - doc-level agreement between the direct citation set and the LOO citation set per item
    (Jaccard, and precision/recall of direct vs the LOO union).
"""
import json, os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "camera_ready_outputs", "T1")
sample = {r["item_id"]: r for r in (json.loads(l) for l in open(os.path.join(OUT, "t1_sample.jsonl"), encoding="utf-8"))}
dpath = os.path.join(OUT, "t1_direct.jsonl")
if not os.path.exists(dpath):
    raise SystemExit("run t1_generate.py first (t1_direct.jsonl missing)")
direct = {r["item_id"]: r for r in (json.loads(l) for l in open(dpath, encoding="utf-8"))}

items = [iid for iid in sample if iid in direct]
print(f"items compared = {len(items)} (of {len(sample)} sampled)")

def sg_map(it):
    return {dc["doc_id"]: dc["self_gen"] for dc in it["docs"]}

# ---- context self-gen share (shared denominator) ----
ctx_sg = ctx_tot = 0
for iid in items:
    for dc in sample[iid]["docs"]:
        ctx_tot += 1; ctx_sg += int(dc["self_gen"])
ctx_share = ctx_sg / ctx_tot

# ---- LOO cited share (pool per-run citations) ----
loo_sg = loo_tot = 0
for iid in items:
    m = sg_map(sample[iid])
    for run in sample[iid]["loo_runs"]:
        for did in run:
            if did in m:
                loo_tot += 1; loo_sg += int(m[did])
loo_share = loo_sg / loo_tot if loo_tot else float("nan")

# ---- direct cited share (one answer per item) ----
dir_sg = dir_tot = 0
for iid in items:
    m = sg_map(sample[iid])
    for did in direct[iid]["cited_ids"]:
        if did in m:
            dir_tot += 1; dir_sg += int(m[did])
dir_share = dir_sg / dir_tot if dir_tot else float("nan")

print("\n================ over-citation ratio, same items ================")
print(f"  context self-gen share      = {ctx_share:.3f}")
print(f"  LOO   cited self-gen share  = {loo_share:.3f}   ratio = {loo_share/ctx_share:.2f}")
print(f"  DIRECT cited self-gen share = {dir_share:.3f}   ratio = {dir_share/ctx_share:.2f}")
print("  -> both >1 and close in magnitude = one over-citation effect measured two ways (C2).")

# ---- doc-level agreement: direct set vs LOO union, per item ----
jac, prec, rec = [], [], []
for iid in items:
    dset = set(direct[iid]["cited_ids"])
    lset = set(did for run in sample[iid]["loo_runs"] for did in run)
    if not dset and not lset:
        continue
    inter = len(dset & lset); union = len(dset | lset)
    jac.append(inter / union if union else 1.0)
    prec.append(inter / len(dset) if dset else float("nan"))
    rec.append(inter / len(lset) if lset else float("nan"))

def m(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")

print("\n================ doc-level agreement (direct vs LOO union, per item) ================")
print(f"  mean Jaccard            = {m(jac):.3f}")
print(f"  mean precision(direct)  = {m(prec):.3f}   (of docs direct cited, share also in LOO)")
print(f"  mean recall(direct)     = {m(rec):.3f}   (of LOO-cited docs, share direct also cited)")
print(f"\n  direct citations/item   = {dir_tot/len(items):.2f}   LOO citations/run-item = {loo_tot/ (len(items)*10):.2f}")
print("### T1 compare done ###")
