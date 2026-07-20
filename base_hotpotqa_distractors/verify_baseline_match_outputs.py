#!/usr/bin/env python3
"""Script 2 — SUCCESS/validity check for the baseline-match sweep outputs.

For every existing arm output, verify it is a COMPLETE, successful run WITHOUT fully loading the
(often >500MB) JSON. Per file it checks:
  - metadata head: num_iterations == expected rounds for the variant (search 30 / replace_one 20 /
    replace_all 10) and num_runs_per_iteration == 10;
  - the file closes cleanly (not truncated mid-write);
  - question count (grep) == 1400;
  - empty-answer rate (grep) <= 10%  (catches all-empty / broken runs, e.g. detok failures).

Run:
    python base_hotpotqa_distractors/verify_baseline_match_outputs.py
"""
import os
import re
import subprocess
import sys

BASE = os.environ.get(
    "BASE_OUTDIR",
    "/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/baseline_match_temp1",
)
MODELS = ["qwen2.5-14b", "llama-3.1-8b", "mistral-7b", "deepseek-r1-distill-qwen-7b"]
VARIANTS = ["search", "replace_one", "replace_all"]
ARMS = {
    "diverse_synth": [("f", x) for x in ("0", "0.3", "0.5", "0.7")],
    "equal_diverse_synth": [("t", x) for x in ("0", "1", "2", "3", "4")],
}
EXP_ROUNDS = {"search": 30, "replace_one": 20, "replace_all": 10}
EXP_Q = 1400
EXP_RUNS = 10
EMPTY_MAX = 0.10


def _head(path, n=8192):
    with open(path, "rb") as f:
        return f.read(n).decode("utf-8", "replace")


def _tail(path, n=256):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(max(0, size - n))
        return f.read().decode("utf-8", "replace")


def _grepc(pattern, path):
    """Count matches of an extended-regex pattern via streaming grep (no full JSON load)."""
    p = subprocess.run(["grep", "-oE", pattern, path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return p.stdout.count(b"\n")


def check(var, path):
    problems = []
    h = _head(path)
    m = re.search(r'"num_iterations":\s*(\d+)', h)
    rounds = int(m.group(1)) if m else None
    m = re.search(r'"num_runs_per_iteration":\s*(\d+)', h)
    runs = int(m.group(1)) if m else None
    if rounds != EXP_ROUNDS[var]:
        problems.append(f"rounds={rounds}!={EXP_ROUNDS[var]}")
    if runs != EXP_RUNS:
        problems.append(f"runs={runs}!={EXP_RUNS}")
    if not _tail(path).rstrip().endswith("}"):
        problems.append("truncated(bad JSON tail)")
    qn = _grepc('"question_id"', path)
    if qn != EXP_Q:
        problems.append(f"questions={qn}!={EXP_Q}")
    total = _grepc('"answer": *"', path)
    empty = _grepc('"answer": *""', path)
    rate = (empty / total) if total else 1.0
    if rate > EMPTY_MAX:
        problems.append(f"empty-answers={rate:.1%}")
    return problems, qn, rate


def main():
    ok = bad = missing = 0
    print(f"results dir: {BASE}\n")
    for model in MODELS:
        for mode, arms in ARMS.items():
            for var in VARIANTS:
                for kind, a in arms:
                    arm = f"{kind}{a}"
                    path = os.path.join(BASE, mode, model, f"base_{var}_{arm}.json")
                    tag = f"{model}/{mode}/{var}_{arm}"
                    if not (os.path.isfile(path) and os.path.getsize(path) > 0):
                        missing += 1
                        continue
                    probs, qn, rate = check(var, path)
                    if probs:
                        bad += 1
                        print(f"FAIL {tag}: {'; '.join(probs)}")
                    else:
                        ok += 1
    print(f"\nOK={ok}  FAIL={bad}  MISSING={missing}  (of {len(MODELS) * 27} expected arms)")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
