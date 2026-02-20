import json
import re
import random
from itertools import combinations
import numpy as np
from llm_service.open_source_llm import EmbeddingModel, OpenSourceLLM


WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
SAME_ANSWER_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SAME_ANSWER_SAMPLE_PAIRS = 10
SAME_ANSWER_SEED = 42


def _tokenize_words(text: str) -> list[str]:
    return [token.lower() for token in WORD_PATTERN.findall(text or "")]


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


def calculate_unique_words(answers: list[str]) -> int:
    unique_words = set()
    for answer in answers:
        unique_words.update(_tokenize_words(answer))
    return len(unique_words)

def _build_same_answer_conversation(answer_a: str, answer_b: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a strict paraphrase judge. "
                "Two answers are the same answer if they express the same core claim(s), "
                "even with different wording or order. "
                "Respond with exactly one token: YES or NO."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Answer A:\n{answer_a}\n\n"
                f"Answer B:\n{answer_b}\n\n"
                "Are these the same answer?"
            ),
        },
    ]


def _parse_yes_no(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if normalized.startswith("yes"):
        return True
    if normalized.startswith("no"):
        return False

    match = re.search(r"\b(yes|no)\b", normalized)
    if match:
        return match.group(1) == "yes"
    return False


def _normalize_generation(gen) -> str:
    if isinstance(gen, str):
        return gen
    if isinstance(gen, dict):
        return gen.get("response") or gen.get("text") or gen.get("content") or str(gen)
    return str(gen)


def _judge_pairs_batch(judge_llm, judge_conversations: list[list[dict[str, str]]]) -> list[str]:
    outputs = judge_llm.inference_batch(judge_conversations)
    return [_normalize_generation(o) for o in outputs]


def calculate_same_answer_percentage(
    answers: list[str],
    judge_llm,
    sample_pairs: int,
    rng: random.Random,
) -> float:
    pairs = list(combinations(range(len(answers)), 2))
    if not pairs:
        return 0.0

    sampled_pairs = pairs
    if sample_pairs > 0 and sample_pairs < len(pairs):
        sampled_pairs = rng.sample(pairs, sample_pairs)

    judge_conversations = [
        _build_same_answer_conversation(answers[i], answers[j])
        for i, j in sampled_pairs
    ]
    judge_outputs = _judge_pairs_batch(judge_llm, judge_conversations)
    same_count = sum(1 for output in judge_outputs if _parse_yes_no(output))
    return 100.0 * same_count / len(sampled_pairs)


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
    rng = random.Random(SAME_ANSWER_SEED)

    judge_llm = OpenSourceLLM(
        model_name=SAME_ANSWER_MODEL_NAME,
        temperature=0.0,
        max_tokens=8,
        top_p=1.0,
        cache_dir=cache_dir,
        disable_log_stats=True,
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
            pairwise_metrics = calculate_pairwise_similarities(embeddings)
            unique_words = calculate_unique_words(answers)

            metrics = {
                **pairwise_metrics,
                "unique_words": unique_words,
            }
            metrics["same_answer_percentage"] = float(
                calculate_same_answer_percentage(
                    answers=answers,
                    judge_llm=judge_llm,
                    sample_pairs=SAME_ANSWER_SAMPLE_PAIRS,
                    rng=rng,
                )
            )

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
            "same_answer_judge_enabled": True,
            "same_answer_judge_model_name": SAME_ANSWER_MODEL_NAME,
            "same_answer_sample_pairs": SAME_ANSWER_SAMPLE_PAIRS,
            "same_answer_seed": SAME_ANSWER_SEED,
        },
        "questions": questions_results,
        #hardcoded, need to update
        "aggregate_statistics": {
            "avg_collapse_rate": 1,
        },
    }

    embed_model.shutdown()
    try:
        judge_llm.shutdown()
    except Exception:
        pass

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

