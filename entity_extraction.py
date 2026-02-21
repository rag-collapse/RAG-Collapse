"""
Entity Extraction and Analysis for RAG Collapse Experiments.

Extracts entities from experiment responses, clusters them into canonical forms,
identifies missed matches, and computes entity-based metrics:
  - Unique entities per round
  - Entity similarity (cosine similarity of binary entity mention vectors)

Usage:
    python entity_extraction.py \
        --model-mode api \
        --model-name gpt-4o \
        --experiment-files experiment_outputs/local_search.json \
                          experiment_outputs/local_replace_one.json \
                          experiment_outputs/local_replace_all.json \
        --output-dir entity_extraction_output
"""

import argparse
import json
import os
import re
import numpy as np
from typing import Any, Dict, List, Set
from tqdm import tqdm

from pipeline.model_runner import build_llm
from llm_service.common_llm import CommonLLM


# ─── Prompts ────────────────────────────────────────────────────────────────────

ENTITY_EXTRACTION_SYSTEM_PROMPT = (
    "You are an entity extraction assistant. Given a question and a response, "
    "extract ONLY the specific entities that are being listed or compared as "
    "answers to the question.\n\n"
    "Important guidelines:\n"
    "- Extract only entities that directly answer the question (e.g., for "
    "\"What are the best restaurants in California?\", extract restaurant names, "
    "NOT city names or chef names).\n"
    "- Each entity should be a specific named entity (person, place, organization, "
    "product, etc.).\n"
    "- Return your answer as a JSON array of strings.\n"
    "- If no relevant entities are found, return an empty array [].\n"
    "- Do NOT include generic descriptions or categories as entities."
)

ENTITY_EXTRACTION_USER_PROMPT = (
    "Question: {question}\n\n"
    "Response: {response}\n\n"
    "Extract the specific entities being listed or compared as answers to the "
    "question. Return ONLY a JSON array of entity name strings, nothing else.\n\n"
    "Example format: [\"Entity 1\", \"Entity 2\", \"Entity 3\"]"
)

ENTITY_CLUSTERING_SYSTEM_PROMPT = (
    "You are an entity resolution assistant. Given a list of entity mentions, "
    "group them into clusters where each cluster contains mentions that refer "
    "to the same real-world entity.\n\n"
    "Important guidelines:\n"
    "- Account for name variations (e.g., \"Charli D'Amelio\" and \"Charli\" "
    "referring to the same person).\n"
    "- Account for typos and misspellings.\n"
    "- Account for abbreviations and acronyms.\n"
    "- Choose the most complete/canonical name as the cluster key.\n"
    "- Return your answer as a JSON object mapping canonical names to arrays "
    "of mentions.\n"
    "- Every input mention must appear in exactly one cluster."
)

ENTITY_CLUSTERING_USER_PROMPT = (
    'Here are all unique entity mentions found across responses to the question: '
    '"{question}"\n\n'
    "Entity mentions:\n{entities}\n\n"
    "Group these mentions into clusters of the same entity. Return ONLY a JSON "
    "object mapping canonical entity names to arrays of all mentions that refer "
    "to that entity.\n\n"
    'Example format:\n'
    '{{"Full Name Person": ["Full Name Person", "First Name", "Last Name"], '
    '"Another Entity": ["Another Entity", "Alt Name"]}}'
)


# ─── Helpers ────────────────────────────────────────────────────────────────────

def parse_json_from_response(response: str) -> Any:
    """Extract JSON from an LLM response, handling markdown code blocks."""
    # Try code blocks first
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try the raw response
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try to locate an array
    arr = re.search(r"\[.*\]", response, re.DOTALL)
    if arr:
        try:
            return json.loads(arr.group(0))
        except json.JSONDecodeError:
            pass

    # Try to locate an object
    obj = re.search(r"\{.*\}", response, re.DOTALL)
    if obj:
        try:
            return json.loads(obj.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ─── Core algorithm ─────────────────────────────────────────────────────────────

def extract_entities_batch(
    llm: CommonLLM,
    question: str,
    responses: List[str],
    batch_size: int = 20,
) -> List[List[str]]:
    """Use the LLM to extract answer-relevant entities from each response."""
    all_entities: List[List[str]] = []

    for i in range(0, len(responses), batch_size):
        batch = responses[i : i + batch_size]
        conversations = [
            [
                {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ENTITY_EXTRACTION_USER_PROMPT.format(
                        question=question, response=resp
                    ),
                },
            ]
            for resp in batch
        ]

        llm_outputs = llm.inference_batch(conversations)

        for output in llm_outputs:
            parsed = parse_json_from_response(output)
            if isinstance(parsed, list):
                entities = [str(e).strip() for e in parsed if e and str(e).strip()]
                all_entities.append(entities)
            else:
                all_entities.append([])

    return all_entities


def cluster_entities(
    llm: CommonLLM,
    question: str,
    unique_mentions: List[str],
) -> Dict[str, List[str]]:
    """Use the LLM to cluster entity mentions into canonical groups."""
    if not unique_mentions:
        return {}

    entities_str = "\n".join(f"- {m}" for m in unique_mentions)

    conversation = [
        {"role": "system", "content": ENTITY_CLUSTERING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": ENTITY_CLUSTERING_USER_PROMPT.format(
                question=question, entities=entities_str
            ),
        },
    ]

    outputs = llm.inference_batch([conversation])
    parsed = parse_json_from_response(outputs[0])

    if isinstance(parsed, dict):
        canonical_map: Dict[str, List[str]] = {}
        for key, mentions in parsed.items():
            if isinstance(mentions, list):
                canonical_map[str(key)] = [str(m) for m in mentions]
            else:
                canonical_map[str(key)] = [str(mentions)]
        return canonical_map

    # Fallback: every mention is its own canonical entity
    return {m: [m] for m in unique_mentions}


def build_mention_to_canonical(canonical_map: Dict[str, List[str]]) -> Dict[str, str]:
    """Reverse-map: lowercased mention -> canonical name."""
    m2c: Dict[str, str] = {}
    for canonical, mentions in canonical_map.items():
        for mention in mentions:
            m2c[mention.lower()] = canonical
    return m2c


def recover_missed_entities(
    response: str,
    extracted_entities: List[str],
    mention_to_canonical: Dict[str, str],
) -> List[str]:
    """Scan the response for known mentions that the LLM missed."""
    response_lower = response.lower()

    # Canonical entities already covered
    already_found: Set[str] = set()
    for entity in extracted_entities:
        canonical = mention_to_canonical.get(entity.lower())
        if canonical:
            already_found.add(canonical)

    recovered: Set[str] = set()
    for mention, canonical in mention_to_canonical.items():
        if canonical not in already_found and mention in response_lower:
            recovered.add(canonical)

    return sorted(recovered)


# ─── Metrics ────────────────────────────────────────────────────────────────────

def compute_entity_similarity(entity_vectors: List[np.ndarray]) -> Dict[str, float]:
    """Pairwise cosine similarity between binary entity-mention vectors."""
    n = len(entity_vectors)
    if n < 2:
        return {
            "mean_entity_similarity": 0.0,
            "min_entity_similarity": 0.0,
            "max_entity_similarity": 0.0,
            "std_entity_similarity": 0.0,
        }

    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            dot = np.dot(entity_vectors[i], entity_vectors[j])
            n1 = np.linalg.norm(entity_vectors[i])
            n2 = np.linalg.norm(entity_vectors[j])
            sim = float(dot / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0
            similarities.append(sim)

    sims = np.array(similarities)
    return {
        "mean_entity_similarity": float(np.mean(sims)),
        "min_entity_similarity": float(np.min(sims)),
        "max_entity_similarity": float(np.max(sims)),
        "std_entity_similarity": float(np.std(sims)),
    }


# ─── Main processing ────────────────────────────────────────────────────────────

def process_experiment_file(
    llm: CommonLLM,
    experiment_file: str,
    output_file: str,
    model_name: str,
    batch_size: int = 20,
) -> dict:
    """Process one experiment file: extract, cluster, recover, compute metrics."""
    with open(experiment_file, "r", encoding="utf-8") as f:
        experiment_data = json.load(f)

    questions_results = []

    for question_data in tqdm(experiment_data["questions"], desc="Questions"):
        question_text = question_data["question_text"]
        question_id = question_data["question_id"]

        # ── Step 1: extract entities from every response ──
        all_responses: List[str] = []
        response_index_map: List[tuple] = []  # (iter_idx, run_idx)

        for iter_idx, iteration in enumerate(question_data["iterations"]):
            for run_idx, run in enumerate(iteration["runs"]):
                all_responses.append(run["answer"])
                response_index_map.append((iter_idx, run_idx))

        print(
            f"  Extracting entities from {len(all_responses)} responses "
            f"for question {question_id} ..."
        )
        all_entities = extract_entities_batch(
            llm, question_text, all_responses, batch_size
        )

        # Organise by iteration
        iterations_entities: Dict[int, Dict[int, List[str]]] = {}
        for (iter_idx, run_idx), entities in zip(response_index_map, all_entities):
            iterations_entities.setdefault(iter_idx, {})[run_idx] = entities

        # ── Step 2: cluster all unique mentions ──
        all_unique_mentions: Set[str] = set()
        for entities_list in all_entities:
            all_unique_mentions.update(entities_list)

        sorted_mentions = sorted(all_unique_mentions)
        print(
            f"  Found {len(sorted_mentions)} unique entity mentions. Clustering ..."
        )
        canonical_map = cluster_entities(llm, question_text, sorted_mentions)
        mention_to_canonical = build_mention_to_canonical(canonical_map)

        all_canonical_entities = sorted(canonical_map.keys())
        canonical_to_idx = {
            name: idx for idx, name in enumerate(all_canonical_entities)
        }
        num_canonical = len(all_canonical_entities)

        # ── Step 3: recover missed matches & compute metrics per iteration ──
        iterations_results = []

        for iter_idx, iteration in enumerate(question_data["iterations"]):
            runs_data = []
            entity_vectors: List[np.ndarray] = []
            canonical_in_round: Set[str] = set()

            for run_idx, run in enumerate(iteration["runs"]):
                response = run["answer"]
                extracted = iterations_entities.get(iter_idx, {}).get(run_idx, [])

                # Map to canonical
                canonical_for_run: Set[str] = set()
                for entity in extracted:
                    canonical = mention_to_canonical.get(entity.lower())
                    if canonical:
                        canonical_for_run.add(canonical)
                    else:
                        canonical_for_run.add(entity)

                # Recover missed
                recovered = recover_missed_entities(
                    response, extracted, mention_to_canonical
                )
                canonical_for_run.update(recovered)
                canonical_in_round.update(canonical_for_run)

                # Binary vector
                vec = np.zeros(num_canonical)
                for ce in canonical_for_run:
                    if ce in canonical_to_idx:
                        vec[canonical_to_idx[ce]] = 1.0
                entity_vectors.append(vec)

                runs_data.append(
                    {
                        "run_id": run["run_id"],
                        "extracted_entities": extracted,
                        "canonical_entities": sorted(canonical_for_run),
                        "recovered_entities": recovered,
                    }
                )

            similarity_metrics = compute_entity_similarity(entity_vectors)

            iterations_results.append(
                {
                    "iteration_number": iteration["iteration_number"],
                    "unique_entities": len(canonical_in_round),
                    "metrics": similarity_metrics,
                    "runs": runs_data,
                }
            )

        questions_results.append(
            {
                "question_id": question_id,
                "question_text": question_text,
                "canonical_entity_map": canonical_map,
                "num_canonical_entities": len(all_canonical_entities),
                "iterations": iterations_results,
            }
        )

    # ── Aggregate statistics ──
    max_iterations = (
        max(len(q["iterations"]) for q in questions_results)
        if questions_results
        else 0
    )

    agg_unique: List[float] = []
    agg_similarity: List[float] = []
    for iter_num in range(max_iterations):
        unique_counts: List[int] = []
        sims: List[float] = []
        for q in questions_results:
            if iter_num < len(q["iterations"]):
                unique_counts.append(q["iterations"][iter_num]["unique_entities"])
                sims.append(
                    q["iterations"][iter_num]["metrics"]["mean_entity_similarity"]
                )
        agg_unique.append(float(np.mean(unique_counts)) if unique_counts else 0.0)
        agg_similarity.append(float(np.mean(sims)) if sims else 0.0)

    results = {
        "experiment_metadata": experiment_data["experiment_metadata"],
        "entity_extraction_metadata": {
            "extraction_model": model_name,
            "source_file": os.path.basename(experiment_file),
        },
        "questions": questions_results,
        "aggregate_statistics": {
            "mean_unique_entities_per_round": agg_unique,
            "mean_entity_similarity_per_round": agg_similarity,
        },
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to {output_file}")
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract entities from RAG experiment responses and compute "
            "entity-based metrics (unique entities, entity similarity)."
        )
    )

    parser.add_argument(
        "--experiment-files",
        nargs="+",
        required=True,
        help="Paths to one or more experiment JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="entity_extraction_output",
        help="Directory to save results (default: entity_extraction_output)",
    )
    parser.add_argument(
        "--model-mode",
        type=str,
        choices=["api", "local"],
        default="api",
        help="LLM mode: 'api' for ProprietaryLLM, 'local' for OpenSourceLLM",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name (e.g. gpt-4o for API, Qwen/Qwen2.5-7B-Instruct for local)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0 for deterministic extraction)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens for LLM output (default: 2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for entity extraction LLM calls (default: 20)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Max model context length for local mode (default: 8192)",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.7,
        help="GPU memory utilization for local mode (default: 0.7)",
    )

    args = parser.parse_args()

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    llm, model_name = build_llm(
        model_mode=args.model_mode,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.9,
        require_gpu=(args.model_mode == "local"),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        cuda_visible_devices=cuda_devices,
    )

    print(f"Using LLM: {llm}")

    for experiment_file in args.experiment_files:
        basename = os.path.splitext(os.path.basename(experiment_file))[0]
        output_file = os.path.join(
            args.output_dir, f"{basename}_entity_results.json"
        )
        print(f"\nProcessing: {experiment_file}")
        process_experiment_file(llm, experiment_file, output_file, model_name, args.batch_size)

    print("\nDone!")


if __name__ == "__main__":
    main()
