import json
import numpy as np
from llm_service.open_source_llm import EmbeddingModel
import argparse
import os


def load_experiment_output(filepath: str) -> dict:
    """Load experiment output JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def calculate_pairwise_similarities(embeddings: np.ndarray) -> dict[str, float]:
    """
    Calculate pairwise similarity statistics for a set of embeddings.

    Args:
        embeddings: Array of embeddings, shape (n, embedding_dim)

    Returns:
        Dictionary with avg, max, min, std of pairwise similarities
    """
    n = len(embeddings)
    if n < 2:
        return {
            "avg_pairwise_similarity": 0.0,
            "max_pairwise_similarity": 0.0,
            "min_pairwise_similarity": 0.0,
            "std_pairwise_similarity": 0.0,
        }

    similarities = []
    # Calculate pairwise cosine similarities (dot product for normalized embeddings)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            similarities.append(sim)

    similarities = np.array(similarities)

    return {
        "avg_pairwise_similarity": float(np.mean(similarities)),
        "max_pairwise_similarity": float(np.max(similarities)),
        "min_pairwise_similarity": float(np.min(similarities)),
        "std_pairwise_similarity": float(np.std(similarities)),
    }


def evaluate_experiment(
    experiment_data: dict,
    embedding_model: EmbeddingModel,
    output_json_path: str,
    output_txt_path: str,
):
    """
    Evaluate experiment output by measuring embedding similarities across iterations.

    Args:
        experiment_data: Loaded experiment output data
        embedding_model: EmbeddingModel instance
        output_json_path: Path to save JSON results
        output_txt_path: Path to save text report
    """
    results = {
        "measurement_metadata": {
            "embedding_model": embedding_model.model_name,
            "similarity_metric": "cosine",
        },
        "questions": [],
    }

    for question_data in experiment_data["questions"]:
        question_id = question_data["question_id"]
        question_text = question_data["question_text"]

        question_results = {
            "question_id": question_id,
            "question_text": question_text,
            "iterations": [],
        }

        print(f"Processing Question {question_id}: {question_text}")

        # Process each iteration
        for iteration_data in question_data["iterations"]:
            iteration_num = iteration_data["iteration_number"]

            # Extract all answers from runs
            answers = [run["answer"] for run in iteration_data["runs"]]

            print(f"  Iteration {iteration_num}: {len(answers)} answers")

            # Embed all answers
            embeddings = embedding_model.embed_batch(answers, normalize=True)

            # Calculate pairwise similarities
            metrics = calculate_pairwise_similarities(embeddings)

            question_results["iterations"].append(
                {
                    "iteration_number": iteration_num,
                    "metrics": metrics,
                }
            )

        results["questions"].append(question_results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

    # Save JSON results
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Generate and save text report
    report = generate_text_report(results, experiment_data)
    with open(output_txt_path, "w") as f:
        f.write(report)

    print(f"\nEvaluation complete!")
    print(f"JSON results saved to: {output_json_path}")
    print(f"Text report saved to: {output_txt_path}")


def generate_text_report(results: dict, experiment_data: dict) -> str:
    """Generate a human-readable text report of the evaluation results."""
    lines = []
    lines.append("=" * 80)
    lines.append("MODEL COLLAPSE EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Metadata
    lines.append("EXPERIMENT METADATA:")
    exp_meta = experiment_data.get("experiment_metadata", {})
    lines.append(f"  Model: {exp_meta.get('model', 'N/A')}")
    lines.append(f"  Number of Iterations: {exp_meta.get('num_iterations', 'N/A')}")
    lines.append(
        f"  Runs per Iteration: {exp_meta.get('num_runs_per_iteration', 'N/A')}"
    )
    lines.append("")

    lines.append("MEASUREMENT METADATA:")
    meas_meta = results["measurement_metadata"]
    lines.append(f"  Embedding Model: {meas_meta['embedding_model']}")
    lines.append(f"  Similarity Metric: {meas_meta['similarity_metric']}")
    lines.append("")

    # Per-question results
    for question_result in results["questions"]:
        lines.append(
            f"QUESTION {question_result['question_id']}: {question_result['question_text']}"
        )
        lines.append("-" * 80)

        for iteration in question_result["iterations"]:
            iter_num = iteration["iteration_number"]
            metrics = iteration["metrics"]

            lines.append(f"\n  Iteration {iter_num}:")
            lines.append(
                f"    Average Pairwise Similarity: {metrics['avg_pairwise_similarity']:.4f}"
            )
            lines.append(
                f"    Max Pairwise Similarity:     {metrics['max_pairwise_similarity']:.4f}"
            )
            lines.append(
                f"    Min Pairwise Similarity:     {metrics['min_pairwise_similarity']:.4f}"
            )
            lines.append(
                f"    Std Pairwise Similarity:     {metrics['std_pairwise_similarity']:.4f}"
            )

        # Show progression
        if len(question_result["iterations"]) > 1:
            lines.append("\n  Similarity Progression:")
            for i, iteration in enumerate(question_result["iterations"]):
                avg_sim = iteration["metrics"]["avg_pairwise_similarity"]
                lines.append(f"    Iteration {i}: {avg_sim:.4f}")
                if i > 0:
                    prev_sim = question_result["iterations"][i - 1]["metrics"][
                        "avg_pairwise_similarity"
                    ]
                    change = avg_sim - prev_sim
                    lines.append(f"      Change from previous: {change:+.4f}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model collapse by measuring embedding similarities"
    )
    parser.add_argument(
        "--experiment_output",
        type=str,
        required=True,
        help="Path to experiment output JSON file",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the embedding model to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for models",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="evaluation_outputs/evaluation_results.json",
        help="Path to save JSON evaluation results",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default="evaluation_outputs/evaluation_report.txt",
        help="Path to save text evaluation report",
    )

    args = parser.parse_args()

    # Load experiment data
    print(f"Loading experiment output from: {args.experiment_output}")
    experiment_data = load_experiment_output(args.experiment_output)

    # Initialize embedding model
    print(f"Initializing embedding model: {args.embedding_model}")
    embedding_model = EmbeddingModel(
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )

    # Run evaluation
    print("Running evaluation...")
    evaluate_experiment(
        experiment_data=experiment_data,
        embedding_model=embedding_model,
        output_json_path=args.output_json,
        output_txt_path=args.output_txt,
    )

    # Cleanup
    embedding_model.shutdown()


if __name__ == "__main__":
    main()
