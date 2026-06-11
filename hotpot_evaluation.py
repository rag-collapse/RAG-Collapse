import argparse
import json
import os
import re
import string
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from pipeline.misinfo import contains_entity  # identical matcher used at injection time


def _normalize_answer(text: str) -> str:
    """Lowercase, remove articles/punctuation, collapse whitespace."""
    text = text.lower()
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    return " ".join(text.split())


def _get_tokens(text: str) -> List[str]:
    return _normalize_answer(text).split()


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(ground_truth)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    num_common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, ground_truth: str) -> Tuple[float, float]:
    """Return (em, f1) for a single prediction."""
    prediction = prediction or ""
    return exact_match(prediction, ground_truth), token_f1(prediction, ground_truth)


def load_ground_truth(gt_file: str) -> Dict[str, str]:
    """
    Return {question_id: answer} from a HotPotQA JSON file.

    Accepts two formats:
    - Native HotPotQA: list of {_id, answer, ...}   (e.g. hotpot_dev_fullwiki_v1.json)
    - Flat mapping:    {id: answer, ...}             (produced by build_hotpot_gt.py)
    """
    print(f"[GT] Loading ground-truth from {gt_file} ...")
    with open(gt_file) as f:
        data = json.load(f)

    if isinstance(data, list):
        gt = {row["_id"]: row["answer"] for row in data}
    elif isinstance(data, dict):
        gt = data
    else:
        raise ValueError(f"Unexpected GT file format in {gt_file}")

    print(f"[GT] Loaded {len(gt)} entries.")
    return gt


def _injection_target_string(inj: dict) -> str:
    """The string whose appearance in an answer counts as 'adopted the injected error'.
    Counterfactual/hop → the injected entity; freeform → the judge-discovered asserted answer."""
    if inj.get("mode") == "freeform":
        claims = inj.get("discovered_claims") or {}
        return (claims.get("asserted_answer") or "").strip()
    return (inj.get("injected_entity") or "").strip()


def _injection_metrics_for_iteration(inj: dict, iteration: dict, predictions: List[str]) -> dict:
    """Per-round injection metrics (PoisonedRAG-style retrieval vs generation conditions,
    plus propagation / recovery signals). Reuses contains_entity for all matching."""
    n = len(predictions) or 1
    gold = inj.get("gold_answer") or ""
    target = _injection_target_string(inj)
    implied = (inj.get("implied_answer") or "").strip()

    docs = iteration.get("documents", []) or []
    doc_ids = {d.get("doc_id") for d in docs}
    injected_doc_ids = set(inj.get("injected_doc_ids") or [])
    retrieval_condition = 1.0 if (injected_doc_ids & doc_ids) else 0.0

    injected_match = (
        sum(contains_entity(p, target) for p in predictions) / n if target else None
    )
    gold_match = sum(contains_entity(p, gold) for p in predictions) / n if gold else None
    implied_match = (
        sum(contains_entity(p, implied) for p in predictions) / n if implied else None
    )

    # propagation: injected string present in a synthesized (gen_) doc in this round's context
    doc_propagation = 0.0
    if target:
        for d in docs:
            if str(d.get("doc_id", "")).startswith("gen_") and contains_entity(d.get("text", ""), target):
                doc_propagation = 1.0
                break

    # amplification proxy (string-level, M2): distinct normalized answers this round that are
    # neither gold nor injected nor implied — answer-space divergence beyond the seeded error.
    known = set()
    for v in (gold, target, implied):
        if v:
            known.add(_normalize_answer(v))
    offtarget = {
        _normalize_answer(p) for p in predictions
        if _normalize_answer(p) and not any(contains_entity(p, v) for v in (gold, target, implied) if v)
    }
    amplification_count = len(offtarget)

    return {
        "retrieval_condition": retrieval_condition,
        "injected_match_rate": injected_match,
        "gold_match_rate": gold_match,
        "implied_match_rate": implied_match,
        "doc_propagation": doc_propagation,
        "amplification_count": amplification_count,
    }


def _summarize_injection(per_q_traces: List[dict], inject_round: int) -> dict:
    """Aggregate per-question injection traces into adoption/persistence/recovery stats."""
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return float(sum(xs) / len(xs)) if xs else 0.0

    adopt_rounds, persistences, recovered, adopted_ever = [], [], 0, 0
    for tr in per_q_traces:
        seq = tr["injected_seq"]            # [(it_num, injected_match_rate or None)]
        gold_seq = tr["gold_seq"]
        post = [(it, r) for it, r in seq if it > inject_round and r is not None]
        adopted_iters = [it for it, r in post if r > 0]
        if adopted_iters:
            adopted_ever += 1
            adopt_rounds.append(min(adopted_iters) - inject_round)
            # longest consecutive adopted run
            best = run = 0
            prev = None
            for it, r in post:
                if r and r > 0:
                    run = run + 1 if prev is not None and it == prev + 1 else 1
                    best = max(best, run)
                    prev = it
                else:
                    prev = None
            persistences.append(best)
            final_gold = next((r for it, r in sorted(gold_seq, reverse=True) if r is not None), 0.0)
            if final_gold and final_gold > 0:
                recovered += 1

    n = len(per_q_traces) or 1
    return {
        "n_eligible": len(per_q_traces),
        "adoption_rate": adopted_ever / n,
        "mean_time_to_adoption": (sum(adopt_rounds) / len(adopt_rounds)) if adopt_rounds else None,
        "mean_persistence": (sum(persistences) / len(persistences)) if persistences else 0.0,
        "recovery_rate": (recovered / adopted_ever) if adopted_ever else 0.0,
    }


def evaluate_experiment(
    experiment_file: str,
    output_file: str,
    gt_file: str,
) -> dict:
    with open(experiment_file) as f:
        experiment_data = json.load(f)

    gt = load_ground_truth(gt_file)

    questions_results = []

    # accumulators: iteration_number -> list of metric values across questions
    acc_em: defaultdict = defaultdict(list)
    acc_f1: defaultdict = defaultdict(list)
    acc_max_em: defaultdict = defaultdict(list)
    acc_max_f1: defaultdict = defaultdict(list)

    skipped = 0

    # injection accumulators (populated only when the experiment carries injection records)
    inj_acc = {k: defaultdict(list) for k in (
        "injected_match_rate", "gold_match_rate", "retrieval_condition",
        "doc_propagation", "implied_match_rate")}
    inj_traces: List[dict] = []
    inj_status_counts: defaultdict = defaultdict(int)
    inj_inject_round = None

    for question in experiment_data["questions"]:
        question_id = f"q{question['question_id']}"
        query_id = question.get("query_id", "")
        answer_gt = gt.get(query_id)

        inj = question.get("injection")
        if inj:
            inj_status_counts[inj.get("status", "unknown")] += 1
            if inj_inject_round is None and inj.get("inject_round") is not None:
                inj_inject_round = inj.get("inject_round")

        if answer_gt is None:
            skipped += 1
            # still include the question with null-ish metrics so output is complete
            questions_results.append(
                {
                    "question_id": question_id,
                    "ground_truth_missing": True,
                    "iterations": [],
                }
            )
            continue

        iterations_results = []
        inj_eligible = bool(inj and inj.get("eligible"))
        inj_injected_seq: List = []
        inj_gold_seq: List = []
        for iteration in question["iterations"]:
            it_num = iteration["iteration_number"]
            predictions = [run.get("answer") or "" for run in iteration["runs"]]

            em_scores = [exact_match(p, answer_gt) for p in predictions]
            f1_scores = [token_f1(p, answer_gt) for p in predictions]

            avg_em = float(sum(em_scores) / len(em_scores)) if em_scores else 0.0
            max_em = float(max(em_scores)) if em_scores else 0.0
            avg_f1 = float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0
            max_f1 = float(max(f1_scores)) if f1_scores else 0.0

            metrics = {
                "avg_em": avg_em,
                "max_em": max_em,
                "avg_f1": avg_f1,
                "max_f1": max_f1,
            }

            if inj_eligible:
                inj_m = _injection_metrics_for_iteration(inj, iteration, predictions)
                metrics["injection"] = inj_m
                for k in inj_acc:
                    inj_acc[k][it_num].append(inj_m[k])
                inj_injected_seq.append((it_num, inj_m["injected_match_rate"]))
                inj_gold_seq.append((it_num, inj_m["gold_match_rate"]))

            iterations_results.append(
                {
                    "iteration_number": it_num,
                    "metrics": metrics,
                }
            )

            acc_em[it_num].append(avg_em)
            acc_f1[it_num].append(avg_f1)
            acc_max_em[it_num].append(max_em)
            acc_max_f1[it_num].append(max_f1)

        if inj_eligible:
            inj_traces.append({"injected_seq": inj_injected_seq, "gold_seq": inj_gold_seq})

        questions_results.append(
            {
                "question_id": question_id,
                "iterations": iterations_results,
            }
        )

    if skipped:
        print(
            f"[Warning] {skipped} question(s) had no ground-truth answer and were skipped."
        )

    # aggregate across questions, keyed by iteration number
    all_iters = sorted(set(acc_em.keys()))

    def _mean(lst):
        return float(sum(lst) / len(lst)) if lst else 0.0

    aggregate_statistics = {
        "avg_em_by_iteration": {str(i): _mean(acc_em[i]) for i in all_iters},
        "avg_f1_by_iteration": {str(i): _mean(acc_f1[i]) for i in all_iters},
        "max_em_by_iteration": {str(i): _mean(acc_max_em[i]) for i in all_iters},
        "max_f1_by_iteration": {str(i): _mean(acc_max_f1[i]) for i in all_iters},
    }

    results = {
        "measurement_metadata": {
            "metric": "exact_match_and_f1",
            "normalization": "lowercase, remove articles/punctuation, collapse whitespace",
            "ground_truth_source": gt_file,
        },
        "questions": questions_results,
        "aggregate_statistics": aggregate_statistics,
    }

    if inj_traces:
        def _meanopt(lst):
            vals = [x for x in lst if x is not None]
            return float(sum(vals) / len(vals)) if vals else None

        inj_iters = sorted({i for k in inj_acc for i in inj_acc[k].keys()})
        inj_agg = {
            "asr_by_iteration": {str(i): _meanopt(inj_acc["injected_match_rate"][i]) for i in inj_iters},
            "gold_match_by_iteration": {str(i): _meanopt(inj_acc["gold_match_rate"][i]) for i in inj_iters},
            "retrieval_condition_by_iteration": {str(i): _meanopt(inj_acc["retrieval_condition"][i]) for i in inj_iters},
            "doc_propagation_by_iteration": {str(i): _meanopt(inj_acc["doc_propagation"][i]) for i in inj_iters},
            "implied_match_by_iteration": {str(i): _meanopt(inj_acc["implied_match_rate"][i]) for i in inj_iters},
            "status_counts": dict(inj_status_counts),
        }
        inj_agg.update(_summarize_injection(inj_traces, inj_inject_round or 0))
        results["injection_aggregates"] = inj_agg
        results["measurement_metadata"]["injection_evaluated"] = True
        print(f"[injection] {inj_agg['n_eligible']} eligible | "
              f"adoption_rate={inj_agg['adoption_rate']:.3f} "
              f"recovery_rate={inj_agg['recovery_rate']:.3f} "
              f"statuses={dict(inj_status_counts)}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote evaluation results to {output_file}")
    return results


def save_summary_json(results: dict, summary_file: str):
    """Save a compact JSON with avg EM and F1 per iteration."""
    agg = results["aggregate_statistics"]
    iters = sorted(agg["avg_em_by_iteration"].keys(), key=int)
    summary = [
        {
            "iteration": int(i),
            "avg_em": agg["avg_em_by_iteration"][i],
            "avg_f1": agg["avg_f1_by_iteration"][i],
        }
        for i in iters
    ]
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary JSON to {summary_file}")


def save_f1_plot(results: dict, plot_file: str):
    """Save a line plot of avg F1 per iteration."""
    agg = results["aggregate_statistics"]
    iters = sorted(agg["avg_f1_by_iteration"].keys(), key=int)
    x = [int(i) for i in iters]
    y = [agg["avg_f1_by_iteration"][i] for i in iters]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg F1")
    ax.set_title("Average F1 Score by Iteration")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)
    print(f"Wrote F1 plot to {plot_file}")


def save_injection_plot(results: dict, plot_file: str):
    """Plot per-iteration ASR (injected-match), gold-match, and implied-match."""
    agg = results.get("injection_aggregates")
    if not agg:
        return
    series = [
        ("asr_by_iteration", "Injected-match (ASR)", "o"),
        ("gold_match_by_iteration", "Gold-match", "s"),
        ("implied_match_by_iteration", "Implied-match", "^"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for key, label, marker in series:
        d = agg.get(key, {})
        xs = sorted((int(i) for i, v in d.items() if v is not None))
        if not xs:
            continue
        ax.plot(xs, [d[str(i)] for i in xs], marker=marker, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Rate (fraction of runs)")
    ax.set_title("Injected-error adoption vs. gold recovery by iteration")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)
    print(f"Wrote injection plot to {plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate HotPotQA experiment results with EM and F1 scores."
    )
    parser.add_argument(
        "experiment_file",
        help="Path to the experiment JSON produced by hotpot_pipeline.py",
    )
    parser.add_argument(
        "output_file",
        help="Path to write the evaluation results JSON",
    )
    parser.add_argument(
        "--gt-file",
        required=True,
        help="Path to HotPotQA ground-truth JSON (e.g. hotpot_dev_fullwiki_v1.json)",
    )
    parser.add_argument(
        "--summary-json",
        required=True,
        help="Path to write the summary JSON (avg EM/F1 per iteration)",
    )
    parser.add_argument(
        "--plot-file",
        required=True,
        help="Path to write the avg F1 plot (e.g. f1_plot.png)",
    )
    args = parser.parse_args()

    results = evaluate_experiment(
        experiment_file=args.experiment_file,
        output_file=args.output_file,
        gt_file=args.gt_file,
    )
    save_summary_json(results, args.summary_json)
    save_f1_plot(results, args.plot_file)
    if results.get("injection_aggregates"):
        base, ext = os.path.splitext(args.plot_file)
        save_injection_plot(results, f"{base}_injection{ext or '.png'}")
    print(json.dumps(results["aggregate_statistics"], indent=2))
