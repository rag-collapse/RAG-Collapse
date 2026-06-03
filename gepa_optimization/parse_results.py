"""
Parse GEPA run results and report the best prompt.

Workflow:
  1. Load logs/aggregates.jsonl  — per-candidate, per-question scores
  2. Load logs/prompts.jsonl     — candidate_id -> prompt text mapping
  3. For each candidate, compute mean avg_anti_collapse and mean avg_quality
     across all questions it was evaluated on
  4. Rank by anti_collapse (primary) then quality (secondary)
  5. Print the leaderboard and the best prompt text

Usage:
  python gepa_optimization/parse_results.py --run-dir gepa_runs/rag_system_prompt_57469398
  python gepa_optimization/parse_results.py --run-dir gepa_runs/rag_system_prompt_57469398 --min-evals 10
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to a gepa_runs/<name>/ directory")
    parser.add_argument(
        "--min-evals", type=int, default=1,
        help="Minimum number of question evaluations required to include a candidate (default: 1)"
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    logs_dir = run_dir / "logs"

    # ── Load data ─────────────────────────────────────────────────────────────
    aggregates = load_jsonl(logs_dir / "aggregates.jsonl")
    prompts_raw = load_jsonl(logs_dir / "prompts.jsonl")

    # candidate_id -> prompt text (keep latest entry per id)
    prompts: dict[str, str] = {}
    for p in prompts_raw:
        prompts[p["candidate_id"]] = p["prompt_text"]

    # ── Aggregate scores per candidate ────────────────────────────────────────
    ac_scores: dict[str, list[float]] = defaultdict(list)
    q_scores:  dict[str, list[float]] = defaultdict(list)

    for row in aggregates:
        cid = row["candidate_id"]
        ac_scores[cid].append(row["avg_anti_collapse"])
        q_scores[cid].append(row["avg_quality"])

    # ── Build leaderboard ─────────────────────────────────────────────────────
    rows = []
    for cid, ac_list in ac_scores.items():
        n = len(ac_list)
        if n < args.min_evals:
            continue
        mean_ac = sum(ac_list) / n
        mean_q  = sum(q_scores[cid]) / len(q_scores[cid])
        source = "seed" if prompts_raw[0]["candidate_id"] == cid else "mutation"
        rows.append((cid, mean_ac, mean_q, n, source))

    rows.sort(key=lambda x: (-x[1], -x[2]))

    # ── Print leaderboard ─────────────────────────────────────────────────────
    print(f"\nRun: {run_dir}")
    print(f"Candidates evaluated: {len(rows)}  (min_evals={args.min_evals})\n")
    print(f"{'Rank':<5} {'candidate_id':<14} {'anti_collapse':>13} {'quality':>8} {'n_evals':>8}  source")
    print("-" * 65)
    for rank, (cid, ac, q, n, src) in enumerate(rows, 1):
        marker = " <-- best" if rank == 1 else ""
        print(f"{rank:<5} {cid:<14} {ac:>13.4f} {q:>8.4f} {n:>8}  {src}{marker}")

    # ── Print best prompt ─────────────────────────────────────────────────────
    best_cid, best_ac, best_q, best_n, best_src = rows[0]
    best_prompt = prompts.get(best_cid, "(prompt text not found)")

    print(f"\n{'='*65}")
    print(f"Best candidate : {best_cid}  ({best_src})")
    print(f"anti_collapse  : {best_ac:.4f}  (over {best_n} questions)")
    print(f"quality        : {best_q:.4f}")
    print(f"{'='*65}\n")
    print(best_prompt)
    print()


if __name__ == "__main__":
    main()
