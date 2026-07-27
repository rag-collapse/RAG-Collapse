#!/usr/bin/env python3
"""Compare the base-HotpotQA distractor-fraction sweep arms.

Reads the per-fraction run JSONs (fraction parsed from the filename ``base_<variant>_f<frac>.json``)
and computes, **over the SAME eligible question cohort for every arm**, the per-round diverse-aware
metrics — so the no-distractor baseline (fraction 0) and the distractor arms are directly comparable:

  gold_match           fraction of runs whose answer CONTAINS the gold entity (accuracy / recovery)
  distractor_adoption  fraction adopting ANY of THIS arm's seeded wrong entities (containment)
  offtarget            fraction off-gold AND off-every-seeded-entity (non-empty answers only)
  distinct_answers     mean number of distinct normalized answers per question (answer diversity)

Matching is CONTAINMENT (``contains_entity``), identical to ``hotpot_evaluation`` — the model's full
answer is checked for the gold/seeded entity, so reasoning-model answers (e.g. DeepSeek) are scored
correctly rather than failing an exact-string ``==``. The eligible cohort is the INTERSECTION of the
questions each distractor arm actually seeded (eligible AND >=1 injected entity), so every arm is
scored on the same questions; the no-distractor baseline (seeds nothing) is scored on that cohort too.
Each arm is scored against ITS OWN seeded entities (not a union across arms). Writes a summary JSON and
prints a compact dose-response table.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from pipeline.misinfo import normalize, contains_entity, load_ground_truth


def _fraction_of(path):
    """Arm label parsed from the TRAILING filename suffix: ``_f<frac>`` (fraction sweep) or
    ``_t<topics>`` (equal_diverse_synth topic sweep). Anchored at end-of-name so a ``_t``/``_f``
    elsewhere in a variant/owner token can't mis-parse. Returns the numeric label as a string."""
    base = os.path.basename(path)
    m = re.search(r"_(?:t|f)([0-9.]+)\.json$", base)
    if not m:
        raise ValueError(f"cannot parse arm label (expected trailing _f<frac> or _t<topics>) from: {base}")
    return m.group(1)


def _qid(q):
    return q.get("query_id") or q.get("question_id")


def _answers_by_round(q):
    """Round -> list of RAW answer strings (one per run). Kept raw so ``contains_entity`` can do its
    own normalization for containment; distinct-answer counting normalizes inline."""
    return {it.get("iteration_number"): [r.get("answer", "") or "" for r in it.get("runs", [])]
            for it in q.get("iterations", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="per-fraction run JSONs (fraction parsed from filename)")
    ap.add_argument("--gt-file", required=True)
    ap.add_argument("--summary", required=True, help="output summary JSON path")
    args = ap.parse_args()

    gt = load_ground_truth(args.gt_file)

    arms = {}                 # arm label(str) -> experiment dict
    seeded_by_arm = {}        # arm -> {query_id: {raw seeded wrong entities}}  (PER ARM, not a union)
    elig_by_arm = {}          # arm -> {query_ids this arm actually seeded (eligible AND >=1 entity)}
    for p in args.runs:
        frac = _fraction_of(p)
        d = json.load(open(p))
        arms[frac] = d
        sb = {}
        for q in d.get("questions", []):
            rec = q.get("initial_distractor") or {}
            if not rec.get("eligible"):
                continue
            ents = {x["injected_entity"] for x in rec.get("distractors", []) if x.get("injected_entity")}
            if not ents:      # eligible but nothing seeded (no_docs / no_docs_in_cache) -> not a treated q
                continue
            sb[_qid(q)] = ents
        seeded_by_arm[frac] = sb
        elig_by_arm[frac] = set(sb)

    # Cohort = INTERSECTION of the treated-question sets across the distractor arms (arms that seeded
    # something). The no-distractor baseline seeds nothing, so it does not constrain the cohort but is
    # scored on it. Every arm is thus scored on the SAME questions, each against ITS OWN seeded entities.
    treated_arms = [a for a in arms if elig_by_arm.get(a)]
    cohort = set.intersection(*(elig_by_arm[a] for a in treated_arms)) if treated_arms else set()

    summary = {"gt_file": args.gt_file, "eligible_cohort_size": len(cohort), "fractions": {}}
    for frac in sorted(arms, key=lambda f: float(f)):
        d = arms[frac]
        seeded = seeded_by_arm.get(frac, {})
        qmap = {_qid(q): q for q in d.get("questions", []) if _qid(q) in cohort}
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
                g = gt.get(cid, "")                    # raw gold; contains_entity normalizes internally
                sset = seeded.get(cid, set())          # THIS arm's seeded entities only
                gm += (sum(contains_entity(a, g) for a in abr) / len(abr)) if g else 0.0
                da += (sum(any(contains_entity(a, e) for e in sset) for a in abr) / len(abr)) if sset else 0.0
                off += sum(1 for a in abr
                           if normalize(a)
                           and not (g and contains_entity(a, g))
                           and not any(contains_entity(a, e) for e in sset)) / len(abr)
                dist += len({normalize(a) for a in abr if normalize(a)})
            if nq:
                per_round[str(it)] = {"gold_match": gm / nq, "distractor_adoption": da / nq,
                                      "offtarget": off / nq, "distinct_answers": dist / nq,
                                      "n_questions": nq}
        summary["fractions"][frac] = per_round

    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"eligible cohort: {len(cohort)} questions  (intersection; metrics computed over it for ALL arms)")
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
