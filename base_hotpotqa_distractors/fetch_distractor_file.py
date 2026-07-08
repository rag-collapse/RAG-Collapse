#!/usr/bin/env python
"""Generate the HotpotQA paper distractor-setting dev file (``hotpot_dev_distractor_v1.json``).

The original HotpotQA distractor setting (Yang et al., EMNLP 2018) gives each question 2 gold
supporting paragraphs + 8 TF-IDF distractor paragraphs (10 context paragraphs total). The official
host (``curtis.ml.cmu.edu``) is dead, so we reconstruct the *identical* dev data from HuggingFace
(``hotpot_qa``, config ``distractor``, split ``validation`` = the 7,405 dev questions) and write it
in the raw JSON schema the pipeline expects:

    [{"_id", "question", "answer", "type", "level",
      "supporting_facts": [[title, sent_id], ...],
      "context":         [[title, [sentence, ...]], ...]}, ...]

Run on a host with internet + the ragenv env (HF cache via HF_HOME). Usage:
    python fetch_distractor_file.py /path/to/hotpot_dev_distractor_v1.json
"""
import json
import sys


def _convert(ex: dict) -> dict:
    """Convert one HF ``hotpot_qa`` example (parallel-list ``context``/``supporting_facts``)
    into the raw HotpotQA JSON record shape used by the pipeline."""
    sf = ex["supporting_facts"]
    ctx = ex["context"]
    return {
        "_id": ex["id"],
        "question": ex["question"],
        "answer": ex["answer"],
        "type": ex.get("type", ""),
        "level": ex.get("level", ""),
        "supporting_facts": [[t, s] for t, s in zip(sf["title"], sf["sent_id"])],
        "context": [[t, list(s)] for t, s in zip(ctx["title"], ctx["sentences"])],
    }


def generate(out_path: str) -> int:
    from datasets import load_dataset
    try:
        ds = load_dataset("hotpot_qa", "distractor", split="validation")
    except Exception:
        # older datasets versions needed the loading script
        ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    recs = [_convert(ex) for ex in ds]
    with open(out_path, "w") as f:
        json.dump(recs, f)
    return len(recs)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: fetch_distractor_file.py <out_path>")
    n = generate(sys.argv[1])
    print(f"wrote {n} records -> {sys.argv[1]}")
