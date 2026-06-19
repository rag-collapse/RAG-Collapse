#!/usr/bin/env python3
"""Compare the base-HotpotQA distractor-fraction sweep arms.

Reads the per-fraction run JSONs (fraction parsed from the filename ``base_<variant>_f<frac>.json``)
and computes, **over the SAME eligible question cohort for every arm**, the per-round diverse-aware
metrics — so the no-distractor baseline (fraction 0) and the distractor arms are directly comparable:

  gold_match           fraction of runs whose answer == gold (accuracy / recovery)
  distractor_adoption  fraction adopting ANY seeded wrong entity (gold-leak already excluded upstream)
  offtarget            fraction off-gold AND off-every-seeded-entity
  distinct_answers     mean number of distinct normalized answers per question (answer diversity)

The eligible cohort is the set of query_ids that any distractor arm marked eligible (deterministic
across arms under the same seed); the fraction-0 baseline is scored on that same cohort from its raw
answers. Writes a summary JSON and prints a compact dose-response table.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from pipeline.misinfo import normalize, load_ground_truth


def _fraction_of(path):
    base = os.path.basename(path)
    return base.rsplit("_f", 1)[1].rsplit(".json", 1)[0] if "_f" in base else base


def _qid(q):
    return q.get("query_id") or q.get("question_id")


def _answers_by_round(q):
    return {it.get("iteration_number"): [normalize(r.get("answer", "")) for r in it.get("runs", [])]
            for it in q.get("iterations", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="per-fraction run JSONs (fraction parsed from filename)")
    ap.add_argument("--gt-file", required=True)
    ap.add_argument("--summary", required=True, help="output summary JSON path")
    args = ap.parse_args()

    gt = load_ground_truth(args.gt_file)

    arms = {}            # frac(str) -> experiment dict
    eligible = set()     # query_ids eligible for distractors (any arm)
    seeded = {}          # query_id -> {normalized seeded wrong entities}
    for p in args.runs:
        d = json.load(open(p))
        arms[_fraction_of(p)] = d
        for q in d.get("questions", []):
            rec = q.get("initial_distractor")
            if rec and rec.get("eligible"):
                cid = _qid(q)
                eligible.add(cid)
                ents = {normalize(x["injected_entity"]) for x in rec.get("distractors", [])
                        if x.get("injected_entity")}
                seeded.setdefault(cid, set()).update(ents)

    summary = {"gt_file": args.gt_file, "eligible_cohort_size": len(eligible), "fractions": {}}
    for frac in sorted(arms, key=lambda f: float(f)):
        d = arms[frac]
        qmap = {_qid(q): q for q in d.get("questions", []) if _qid(q) in eligible}
        rounds = set()
        for q in qmap.values():
            rounds.update(_answers_by_round(q))
        per_round = {}
        for it in sorted(r for r in rounds if r is not None):
            gm = da = off = dist = 0.0
            nq = 0
            for cid, q in qmap.items():
                abr = _answers_by_round(q).get(it)
                if not abr:
                    continue
                nq += 1
                g = normalize(gt.get(cid, ""))
                sset = seeded.get(cid, set())
                gm += sum(a == g for a in abr) / len(abr)
                da += sum(a in sset for a in abr) / len(abr)
                off += sum(a != g and a not in sset for a in abr) / len(abr)
                dist += len(set(abr))
            if nq:
                per_round[str(it)] = {"gold_match": gm / nq, "distractor_adoption": da / nq,
                                      "offtarget": off / nq, "distinct_answers": dist / nq,
                                      "n_questions": nq}
        summary["fractions"][frac] = per_round

    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"eligible cohort: {len(eligible)} questions  (metrics computed over this cohort for ALL arms)")
    print(f"{'frac':>5} {'r0_gold':>8} {'rF_gold':>8} {'dgold':>7} {'r0_dist':>8} {'rF_dist':>8} "
          f"{'rF_adopt':>9} {'rF_offtgt':>10}")
    base_rf = None
    for frac in sorted(summary["fractions"], key=float):
        pr = summary["fractions"][frac]
        if not pr:
            continue
        its = sorted(pr, key=int)
        r0, rF = pr[its[0]], pr[its[-1]]
        if frac in ("0", "0.0"):
            base_rf = rF["gold_match"]
        dgold = (rF["gold_match"] - base_rf) if base_rf is not None else 0.0
        print(f"{frac:>5} {r0['gold_match']:>8.3f} {rF['gold_match']:>8.3f} {dgold:>7.3f} "
              f"{r0['distinct_answers']:>8.2f} {rF['distinct_answers']:>8.2f} "
              f"{rF['distractor_adoption']:>9.3f} {rF['offtarget']:>10.3f}")
    print(f"\nwrote {args.summary}")


if __name__ == "__main__":
    main()
