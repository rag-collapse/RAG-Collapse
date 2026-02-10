import json
from typing import Any, Dict


def write_experiments_output(
    path: str,
    experiments: Dict[str, Any],
) -> None:
    """
    Write the full experiments output to disk as a single JSON file.
    """
    with open(path, "w") as f:
        json.dump(experiments, f, indent=2)
