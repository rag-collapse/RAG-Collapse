"""
Scoring helpers for GEPA RAG prompt optimization.

anti_collapse_score  — entity-based collapse detection, matching entity_extraction.py.
                       Extracts named entities from each round's answer via the judge
                       LLM, clusters them into canonical forms, then computes:
                         score = 1 - mean_entity_similarity
                       where entity_similarity is pairwise cosine similarity of binary
                       entity-mention vectors across rounds.
                       Higher score = more diverse entity sets = less collapse.
                       Also returns unique_entity_count (total canonical entities
                       mentioned across all rounds).

judge_quality_score  — LLM judges the final-round answer against the ORIGINAL context
                       (not the contaminated one), measuring quality degradation.
"""
import json
import sys
from pathlib import Path
from typing import Tuple

import litellm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from entity_extraction import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_USER_PROMPT,
    ENTITY_CLUSTERING_SYSTEM_PROMPT,
    ENTITY_CLUSTERING_USER_PROMPT,
    parse_json_from_response,
    compute_entity_similarity,
)

ANTI_COLLAPSE_THRESHOLD = 0.3
QUALITY_THRESHOLD = 0.5

_JUDGE_SYSTEM = "You are an answer quality evaluator. Respond with valid JSON only."

_JUDGE_USER = """Question: {question}

Original retrieved context (ground truth):
{context}

Answer produced at the final simulation round:
{answer}

Rate this answer on two dimensions (each 0.0–1.0):
1. Faithfulness: Is every claim supported by the original retrieved context?
   (1.0 = fully grounded, 0.0 = entirely hallucinated or drifted)
2. Relevance: Does the answer directly address the question?
   (1.0 = complete and focused, 0.0 = off-topic or missing)

Respond in JSON only: {{"faithfulness": <float>, "relevance": <float>}}"""


# ── Entity extraction helpers (litellm-backed) ────────────────────────────────

def _extract_entities(
    question: str,
    answers: list[str],
    model: str,
    api_base: str,
    api_key: str,
    _meta: dict | None = None,
) -> list[list[str]]:
    """Extract answer-relevant entities from each answer via litellm."""
    result = []
    call_meta = {**(_meta or {}), "role": "entity_extraction"}
    for answer in answers:
        if not answer or not answer.strip():
            result.append([])
            continue
        messages = [
            {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": ENTITY_EXTRACTION_USER_PROMPT.format(
                question=question, response=answer,
            )},
        ]
        try:
            response = litellm.completion(
                model=model, messages=messages,
                api_base=api_base, api_key=api_key,
                temperature=0.0, max_tokens=256,
                metadata=call_meta,
            )
            raw = response.choices[0].message.content
            parsed = parse_json_from_response(raw)
            if isinstance(parsed, list):
                result.append([str(e).strip() for e in parsed if e and str(e).strip()])
            else:
                result.append([])
        except Exception:
            result.append([])
    return result


def _cluster_entities(
    question: str,
    unique_mentions: list[str],
    model: str,
    api_base: str,
    api_key: str,
    _meta: dict | None = None,
) -> dict[str, list[str]]:
    """Cluster entity mentions into canonical forms via litellm."""
    if not unique_mentions:
        return {}
    entities_str = "\n".join(f"- {m}" for m in unique_mentions)
    messages = [
        {"role": "system", "content": ENTITY_CLUSTERING_SYSTEM_PROMPT},
        {"role": "user", "content": ENTITY_CLUSTERING_USER_PROMPT.format(
            question=question, entities=entities_str,
        )},
    ]
    try:
        response = litellm.completion(
            model=model, messages=messages,
            api_base=api_base, api_key=api_key,
            temperature=0.0, max_tokens=512,
            metadata={**(_meta or {}), "role": "entity_clustering"},
        )
        raw = response.choices[0].message.content
        parsed = parse_json_from_response(raw)
        if isinstance(parsed, dict):
            return {
                str(k): [str(m) for m in v] if isinstance(v, list) else [str(v)]
                for k, v in parsed.items()
            }
    except Exception:
        pass
    return {m: [m] for m in unique_mentions}


# ── Public scoring functions ──────────────────────────────────────────────────

def anti_collapse_score(
    question: str,
    answers: list[str],
    model: str,
    api_base: str,
    api_key: str,
    _meta: dict | None = None,
) -> Tuple[float, int]:
    """
    Returns (score, unique_entity_count).

    score = 1 - mean_entity_similarity, range [0,1].
      Higher = entity sets differ across rounds = less collapse.

    unique_entity_count = total canonical entities mentioned across all rounds.
      Higher = more diverse content = less collapse.

    Falls back to (1.0, 0) when fewer than 2 non-empty answers are available.
    """
    clean = [a for a in answers if a and a.strip()]
    if len(clean) < 2:
        return 1.0, 0

    per_answer_entities = _extract_entities(question, clean, model, api_base, api_key, _meta)

    all_mentions = list({e for entities in per_answer_entities for e in entities})
    if not all_mentions:
        return 1.0, 0

    canonical_map = _cluster_entities(question, all_mentions, model, api_base, api_key, _meta)
    mention_to_canonical = {
        m.lower(): c
        for c, mentions in canonical_map.items()
        for m in mentions
    }
    all_canonicals = list(canonical_map.keys())

    entity_vectors = []
    for entities in per_answer_entities:
        canonical_set = {mention_to_canonical.get(e.lower(), e) for e in entities}
        vec = np.array([1.0 if c in canonical_set else 0.0 for c in all_canonicals])
        entity_vectors.append(vec)

    sim_metrics = compute_entity_similarity(entity_vectors)
    return 1.0 - sim_metrics["mean_entity_similarity"], len(all_canonicals)


def judge_quality_score(
    question: str,
    original_context: str,
    final_answer: str,
    judge_model: str,
    judge_api_base: str,
    judge_api_key: str,
    _meta: dict | None = None,
) -> float:
    """
    Judge how faithful and relevant the final-round answer is against the
    ORIGINAL (uncontaminated) context. Falls back to 0.5 on any error.
    """
    if not final_answer or not final_answer.strip():
        return 0.5
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {
            "role": "user",
            "content": _JUDGE_USER.format(
                question=question,
                context=original_context,
                answer=final_answer,
            ),
        },
    ]
    try:
        response = litellm.completion(
            model=judge_model,
            messages=messages,
            api_base=judge_api_base,
            api_key=judge_api_key,
            temperature=0.0,
            max_tokens=128,
            metadata={**(_meta or {}), "role": "quality_judge"},
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        faithfulness = float(parsed.get("faithfulness", 0.5))
        relevance = float(parsed.get("relevance", 0.5))
        return 0.6 * faithfulness + 0.4 * relevance
    except Exception:
        return 0.5
