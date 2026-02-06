import json
from typing import Any, Dict, List


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
