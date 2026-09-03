"""Score T2 annotations: extractor precision / recall / F1 vs a human, plus exact-match rate.

Usage:
  python t2_score.py <t2_annotations.json> [<annotator2.json> ...]

Each annotations file is what the HTML form's "Export" button produced. With >=2 files it also
reports inter-annotator agreement on the per-entity ✓/✗ marks (Cohen's kappa).

Per item: ok = #entities marked correct, bad = #marked wrong, missed = #entities the annotator
added as missed. precision = ok/(ok+bad); recall = ok/(ok+missed); F1 = harmonic mean. Exact
extraction = (bad==0 and missed==0). Reported overall and split by regime and round stratum.
"""
import json, os, sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "camera_ready_outputs", "T2")
sample = {r["item_id"]: r for r in (json.loads(l) for l in open(os.path.join(OUT, "t2_sample.jsonl"), encoding="utf-8"))}

files = sys.argv[1:]
if not files:
    print("usage: python t2_score.py <t2_annotations.json> [more...]"); sys.exit(1)


def strat(r, source):
    return "early" if r <= 1 else "late" if r >= 27 else "mid"


def score_one(path):
    ann = json.load(open(path, encoding="utf-8"))["annotations"]
    per = {}
    agg = defaultdict(lambda: [0, 0, 0])  # group -> [sum_p, sum_r, n]  (macro)
    micro = [0, 0, 0, 0]  # ok, bad, missed, n
    exact = 0; n = 0
    for iid, a in ann.items():
        if iid not in sample:
            continue
        marks = a.get("marks", {})
        ok = sum(1 for v in marks.values() if v == "ok")
        bad = sum(1 for v in marks.values() if v == "bad")
        missed_txt = (a.get("missed") or "").strip()
        missed = len([m for m in missed_txt.split(",") if m.strip()]) if missed_txt else 0
        if ok + bad + missed == 0:
            continue  # untouched item
        n += 1
        p = ok / (ok + bad) if (ok + bad) else 1.0
        r = ok / (ok + missed) if (ok + missed) else 1.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        if bad == 0 and missed == 0:
            exact += 1
        micro[0] += ok; micro[1] += bad; micro[2] += missed; micro[3] += 1
        it = sample[iid]
        for g in ("all", it["source"], strat(it["round"], it["source"])):
            agg[g][0] += p; agg[g][1] += r; agg[g][2] += 1
        per[iid] = dict(p=p, r=r, f=f, ok=ok, bad=bad, missed=missed)
    return per, agg, micro, exact, n


all_per = {}
for path in files:
    per, agg, micro, exact, n = score_one(path)
    all_per[path] = per
    name = os.path.basename(path)
    print(f"\n==== {name}  (annotated items: {n}) ====")
    ok, bad, missed, _ = micro
    micP = ok / (ok + bad) if (ok + bad) else float("nan")
    micR = ok / (ok + missed) if (ok + missed) else float("nan")
    micF = 2 * micP * micR / (micP + micR) if (micP + micR) else float("nan")
    print(f"  micro  precision={micP:.3f}  recall={micR:.3f}  F1={micF:.3f}   (ok={ok} bad={bad} missed={missed})")
    print(f"  exact-extraction rate = {exact}/{n} = {exact/max(1,n):.3f}")
    print(f"  {'group':10s} {'macro-P':>8s} {'macro-R':>8s} {'n':>4s}")
    for g in ["all", "replace_one", "search", "early", "mid", "late"]:
        if g in agg:
            sp, sr, gn = agg[g]
            print(f"  {g:10s} {sp/gn:8.3f} {sr/gn:8.3f} {gn:4d}")

# inter-annotator agreement (Cohen's kappa on per-entity ok/bad marks) if >=2 files
if len(files) >= 2:
    a0 = json.load(open(files[0]))["annotations"]; a1 = json.load(open(files[1]))["annotations"]
    both = []
    for iid in set(a0) & set(a1):
        m0 = a0[iid].get("marks", {}); m1 = a1[iid].get("marks", {})
        for k in set(m0) & set(m1):
            both.append((m0[k], m1[k]))
    if both:
        n = len(both); po = sum(1 for x, y in both if x == y) / n
        from collections import Counter
        c0 = Counter(x for x, _ in both); c1 = Counter(y for _, y in both)
        pe = sum((c0[l]/n) * (c1[l]/n) for l in set(c0) | set(c1))
        kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0
        print(f"\n==== inter-annotator (first two files) ====")
        print(f"  entities compared={n}  observed agreement={po:.3f}  Cohen's kappa={kappa:.3f}")

print("\n(one annotator: P/R/F1 = extractor-vs-human; add a second export for kappa.)")
