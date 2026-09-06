"""Over-citation ratio from a stored LOO sidecar (no regeneration). For each run|sidecar|label:
ratio = (self-gen share of citations) / (self-gen share of context), pooled over the round's 10 runs
and all questions, reported at round 1 and pooled over all rounds. self-gen tag matches
loo_attribution.py / evaluation.py exactly. Emits a TSV row per pair plus a human summary.

Usage: python overcite_ratio.py "<run.json>|<sidecar.json>|<label>" [ ... ]
"""
import json, sys

def is_ai(did, url, it):
    return did.startswith("gen_") or url == "model_generated" or (isinstance(it, int) and it > 0 and not did.startswith("ref_"))

def share(sg, n):
    return (sg / n) if n else float("nan")

print("label\tr1_ctx%\tr1_cited%\tr1_ratio\tpool_ctx%\tpool_cited%\tpool_ratio\tr1_cites\tpool_cites")
for arg in sys.argv[1:]:
    run_path, sc_path, label = arg.split("|", 2)
    run = json.load(open(run_path))
    sc = json.load(open(sc_path))
    # accumulators: round-1 and pooled-over-all-rounds
    r1_csg = r1_cn = r1_xsg = r1_xn = 0
    p_csg = p_cn = p_xsg = p_xn = 0
    for qi, q in enumerate(run["questions"]):
        qsc = sc.get(str(qi), {})
        for it in q["iterations"]:
            r = it["iteration_number"]
            if r == 0:
                continue
            docs = it.get("documents") or []
            meta = {x["doc_id"]: (x.get("url"), x.get("iteration")) for x in docs if x.get("doc_id")}
            # context share
            for did, (url, itr) in meta.items():
                ai = int(is_ai(did, url, itr))
                p_xn += 1; p_xsg += ai
                if r == 1:
                    r1_xn += 1; r1_xsg += ai
            # citation share (pool across the round's runs)
            rsc = qsc.get(str(r), {})
            for run_id, cids in rsc.items():
                for did in cids:
                    url, itr = meta.get(did, (None, None))
                    ai = int(is_ai(did, url, itr))
                    p_cn += 1; p_csg += ai
                    if r == 1:
                        r1_cn += 1; r1_csg += ai
    r1_ctx, r1_cit = share(r1_xsg, r1_xn), share(r1_csg, r1_cn)
    p_ctx, p_cit = share(p_xsg, p_xn), share(p_csg, p_cn)
    r1_ratio = (r1_cit / r1_ctx) if (r1_ctx and r1_ctx == r1_ctx) else float("nan")
    p_ratio = (p_cit / p_ctx) if (p_ctx and p_ctx == p_ctx) else float("nan")
    print(f"{label}\t{100*r1_ctx:.1f}\t{100*r1_cit:.1f}\t{r1_ratio:.2f}\t{100*p_ctx:.1f}\t{100*p_cit:.1f}\t{p_ratio:.2f}\t{r1_cn}\t{p_cn}")
