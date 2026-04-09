import argparse
import json
import re
import string
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


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

    for question in experiment_data["questions"]:
        question_id = f"q{question['question_id']}"
        query_id = question.get("query_id", "")
        answer_gt = gt.get(query_id)

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
    print(json.dumps(results["aggregate_statistics"], indent=2))
