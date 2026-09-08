"""
Search setting: self (gen_*) vs. original (ref_*) citation rate, at the source-doc
level (Search retrieves multiple chunks per source doc that share a doc_id, so we
aggregate citations by doc_id per round before computing rates). Detector URL split
is unavailable for Search (chunk url fields are empty), so originals are combined.
"""
import json, sys
import numpy as np

def main(path):
    d = json.load(open(path))
    qs = d["questions"]
    print(f"File: {path.split('/')[-1]}  | {len(qs)} questions; {d['experiment_metadata']['num_iterations']} iters")
    per_round = {}  # round -> {'self':[rates], 'orig':[rates]}
    for q in qs:
        for it in q["iterations"]:
            rnd = it["iteration_number"]
            # citations by doc_id, aggregated over runs
            cc, tot = {}, 0
            for run in it["runs"]:
                for c in (run.get("citations") or []):
                    cc[c["doc_id"]] = cc.get(c["doc_id"], 0) + 1
                    tot += 1
            if tot == 0:
                continue
            # unique source docs present this round
            present = {}
            for doc in it["documents"]:
                present[doc["doc_id"]] = "self" if str(doc["doc_id"]).startswith("gen_") else "orig"
            pr = per_round.setdefault(rnd, {"self": [], "orig": []})
            for doc_id, grp in present.items():
                pr[grp].append(cc.get(doc_id, 0) / tot)

    print(f"\n{'rnd':>3} {'cr_self':>8} {'cr_orig':>8} {'ratio':>6} {'n_self':>7} {'n_orig':>7}")
    alls, allo = [], []
    for rnd in sorted(per_round):
        s, o = per_round[rnd]["self"], per_round[rnd]["orig"]
        ms = np.mean(s) if s else float("nan")
        mo = np.mean(o) if o else float("nan")
        ratio = ms/mo if (o and mo>0) else float("nan")
        print(f"{rnd:>3} {ms:>8.3f} {mo:>8.3f} {ratio:>6.2f} {len(s):>7} {len(o):>7}")
        if rnd >= 1:
            alls += s; allo += o

    def ci(v):
        a=np.array(v,float); rng=np.random.default_rng(0)
        b=[a[rng.integers(0,len(a),len(a))].mean() for _ in range(1000)]
        return a.mean(), np.percentile(b,2.5), np.percentile(b,97.5)
    ms,lo,hi = ci(alls); mo,lo2,hi2 = ci(allo)
    print(f"\nPooled over rounds>=1 (originals always present in Search):")
    print(f"  self     n={len(alls):<6} mean={ms:.4f} CI [{lo:.4f},{hi:.4f}]")
    print(f"  original n={len(allo):<6} mean={mo:.4f} CI [{lo2:.4f},{hi2:.4f}]")
    print(f"  self/original ratio = {ms/mo:.2f}x")

if __name__ == "__main__":
    main(sys.argv[1])
