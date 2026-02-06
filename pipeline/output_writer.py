import json
from typing import List, Dict


def write_iteration_output(
    path: str,
    iteration: int,
    records: List[Dict],
):
    """
    Append iteration outputs to disk.

    This function does not evaluate or transform outputs.
    """
    output = {
        "iteration": iteration,
        "records": records,
    }

    with open(path, "a") as f:
        f.write(json.dumps(output) + "\n")
