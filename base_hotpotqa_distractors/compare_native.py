#!/usr/bin/env python3
"""Compare the ORIGINAL HotpotQA distractor-setting run vs the gold-only contrast on QA
accuracy over rounds.

The native distractors are answer-ABSENT (no seeded wrong entity), so there is no adoption
metric — the read is whether accuracy degrades across rounds and how the distractor setting
compares to gold-only. Reads run JSONs (arm label parsed from the filename
``native_<variant>_<arm>.json``), and over the SHARED question cohort computes per round:

  gold_match        fraction of runs whose normalized answer == gold
  distinct_answers  mean number of distinct normalized answers per question (diversity)

Writes a summary JSON and prints a compact table.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from pipeline.misinfo import normalize, load_ground_truth


def _arm(path):
    return os.path.basename(path).rsplit(".json", 1)[0].split("_")[-1]   # ..._distractor / ..._goldonly


def _qid(q):
    return q.get("query_id") or q.get("question_id")


def _by_round(q):
    return {it.get("iteration_number"): [normalize(r.get("answer", "")) for r in it.get("runs", [])]
            for it in q.get("iterations", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="native run JSONs (arm parsed from filename)")
    ap.add_argument("--gt-file", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    gt = load_ground_truth(args.gt_file)
    arms = {_arm(p): json.load(open(p)) for p in args.runs}

    # shared cohort = query_ids present in EVERY arm (apples-to-apples)
    idsets = [{_qid(q) for q in d.get("questions", [])} for d in arms.values()]
    cohort = set.intersection(*idsets) if idsets else set()

    summary = {"gt_file": args.gt_file, "cohort_size": len(cohort), "arms": {}}
    for arm, d in arms.items():
        qmap = {_qid(q): q for q in d.get("questions", []) if _qid(q) in cohort}
        rounds = set()
        for q in qmap.values():
            rounds.update(_by_round(q))
        per_round = {}
        for it in sorted(r for r in rounds if r is not None):
            gm = dist = 0.0
            nq = 0
            for cid, q in qmap.items():
                abr = _by_round(q).get(it)
                if not abr:
                    continue
                nq += 1
                g = normalize(gt.get(cid, ""))
                gm += sum(a == g for a in abr) / len(abr)
                dist += len(set(abr))
            if nq:
                per_round[str(it)] = {"gold_match": gm / nq, "distinct_answers": dist / nq, "n_questions": nq}
        summary["arms"][arm] = per_round

    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"shared cohort: {len(cohort)} questions  (metrics over this cohort for all arms)")
    print(f"{'arm':>12} {'r0_gold':>8} {'rF_gold':>8} {'dgold':>7} {'r0_dist':>8} {'rF_dist':>8}")
    for arm in sorted(summary["arms"]):
        pr = summary["arms"][arm]
        if not pr:
            continue
        its = sorted(pr, key=int)
        r0, rF = pr[its[0]], pr[its[-1]]
        print(f"{arm:>12} {r0['gold_match']:>8.3f} {rF['gold_match']:>8.3f} "
              f"{rF['gold_match'] - r0['gold_match']:>7.3f} "
              f"{r0['distinct_answers']:>8.2f} {rF['distinct_answers']:>8.2f}")
    print(f"\nwrote {args.summary}")


if __name__ == "__main__":
    main()
