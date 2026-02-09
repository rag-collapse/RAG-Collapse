import json
import numpy as np
from llm_service.open_source_llm import EmbeddingModel

def calculate_pairwise_similarities(embeddings: np.ndarray) -> dict:
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
    experiment_file: str,
    output_file: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    cache_dir: str = None,
) -> dict:
    with open(experiment_file, "r") as f:
        experiment_data = json.load(f)

    embed_model = EmbeddingModel(
        model_name=embedding_model_name,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )

    questions_results = []

    for question in experiment_data["questions"]:
        question_id = f"q{question['question_id']}"
        iterations_results = []

        for iteration in question["iterations"]:
            # get all the answers for this iteration
            answers = [run["answer"] for run in iteration["runs"]]
            # compute embeddings for all answers in this iteration
            embeddings = embed_model.embed_batch(answers, normalize=True)
            # compute pairwise similarity metrics for this iteration
            metrics = calculate_pairwise_similarities(embeddings)
            iterations_results.append({
                "iteration_number": iteration["iteration_number"],
                "metrics": metrics,
            })


        questions_results.append({
            "question_id": question_id,
            "iterations": iterations_results,
        })

    # add results metadata and aggregate statistics
    results = {
        "measurement_metadata": {
            "embedding_model": embedding_model_name,
            "similarity_metric": "cosine",
        },
        "questions": questions_results,
        "aggregate_statistics": {
            "avg_collapse_rate": 1,
        },
    }

    embed_model.shutdown()

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate RAG experiment results by calculating pairwise similarities between generated answers."
    )

    parser.add_argument(
        "experiment_file",
        type=str,
        help="Path to the experiment JSON file containing questions and iterations"
    )

    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the evaluation results JSON file"
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the embedding model to use (default: all-MiniLM-L6-v2)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation (default: 32)"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache the embedding model (optional)"
    )

    args = parser.parse_args()

    evaluation_results = evaluate_experiment(
        experiment_file=args.experiment_file,
        output_file=args.output_file,
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )

    print(json.dumps(evaluation_results, indent=2))

