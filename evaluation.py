import json
import re
import random
from itertools import combinations
import numpy as np
from llm_service.open_source_llm import EmbeddingModel, OpenSourceLLM


WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
CITATION_PATTERN = re.compile(r"\[(\d+)\]")
SAME_ANSWER_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
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


def _is_ai_generated_doc(doc: dict) -> bool:
    doc_id = str(doc.get("doc_id", doc.get("document_id", ""))).lower()
    url = str(doc.get("url", "")).lower()
    source = str(doc.get("source", "")).lower()
    return (
        doc_id.startswith("gen_")
        or url == "model_generated"
        or "ai" in source
        or "generated" in source
    )


def _build_doc_lookups(iteration: dict) -> tuple[dict[int, dict], dict[str, dict]]:
    docs = iteration.get("documents", [])
    index_to_doc = {}
    id_to_doc = {}

    if not isinstance(docs, list):
        return index_to_doc, id_to_doc

    for idx, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        index_to_doc[idx] = doc
        doc_id = doc.get("doc_id", doc.get("document_id"))
        if doc_id is not None:
            id_to_doc[str(doc_id)] = doc

    return index_to_doc, id_to_doc


def calculate_ai_citation_percentage(iteration: dict) -> tuple[float, str]:
    index_to_doc, id_to_doc = _build_doc_lookups(iteration)

    total_citations = 0
    ai_citations = 0
    source = "none"

    for run in iteration.get("runs", []):
        if not isinstance(run, dict):
            continue

        structured_citations = run.get("citations")
        if isinstance(structured_citations, list) and structured_citations:
            source = "run.citations"
            for citation in structured_citations:
                if not isinstance(citation, dict):
                    continue
                total_citations += 1
                ref_doc = None
                citation_doc_id = citation.get("doc_id", citation.get("document_id"))
                if citation_doc_id is not None:
                    ref_doc = id_to_doc.get(str(citation_doc_id))
                if ref_doc is None:
                    ref_doc = citation
                if _is_ai_generated_doc(ref_doc):
                    ai_citations += 1
            continue

        answer = run.get("answer", "")
        indices = [int(m.group(1)) for m in CITATION_PATTERN.finditer(answer)]
        if indices:
            source = "answer_brackets"
        for citation_index in indices:
            total_citations += 1
            # Prompt format uses "context 0", so [0] maps to documents[0].
            ref_doc = index_to_doc.get(citation_index)
            if ref_doc is not None and _is_ai_generated_doc(ref_doc):
                ai_citations += 1

    if total_citations == 0:
        return 0.0, source
    return (100.0 * ai_citations / total_citations), source


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
            ai_citation_percentage, ai_citation_source = calculate_ai_citation_percentage(iteration)

            metrics = {
                **pairwise_metrics,
                "unique_words": unique_words,
                "ai_citation_percentage": float(ai_citation_percentage),
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
                "metric_metadata": {
                    "ai_citation_source": ai_citation_source,
                },
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

