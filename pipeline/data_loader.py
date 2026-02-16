import json
from typing import Any, Dict, List

from pipeline.config import MIN_CITATIONS, MAX_CITATIONS_REPLACE, should_truncate_citations


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load a dataset from JSONL or JSON.

    JSONL: one JSON object per line
    JSON:  a single JSON array of objects

    Each record should look like:
      {
        "question": "...",
        "references": [{"url": "...", "text": "..."}]
      }
    """
    if path.endswith(".jsonl"):
        data: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num} of {path}") from e
        return data

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError(f"{path} must contain a JSON array (list) of records")
        return obj

    raise ValueError(f"Unsupported dataset format: {path}. Expected .jsonl or .json")


def filter_questions(
    rows: List[Dict[str, Any]],
    min_citations: int = MIN_CITATIONS,
) -> List[Dict[str, Any]]:
    """Keep only questions that have at least min_citations references."""
    return [r for r in rows if len(r.get("references") or []) >= min_citations]


def truncate_references(
    row: Dict[str, Any],
    max_refs: int = MAX_CITATIONS_REPLACE,
) -> Dict[str, Any]:
    """Return a copy of the row with references truncated to max_refs."""
    refs = (row.get("references") or [])[:max_refs]
    return {**row, "references": refs}


def prepare_dataset(
    rows: List[Dict[str, Any]],
    pipeline_variant: str,
) -> List[Dict[str, Any]]:
    """
    Filter (min citations) and optionally truncate references for the pipeline variant.
    """
    out = filter_questions(rows)
    if should_truncate_citations(pipeline_variant):
        out = [truncate_references(r) for r in out]
    return out
