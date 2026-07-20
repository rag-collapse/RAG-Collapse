#!/usr/bin/env python3
"""Script 1 — EXISTENCE check for the baseline-match sweep outputs.

Enumerates the full matrix (4 models x 2 modes x 3 variants x fractions/topics = 108 arms) and
reports which output files are present vs missing under the /work results dir. Missing = the arm
hasn't finished yet (still running/queued) or failed. Fast: only stat()s files.

Run (from a login node, or pipe over ssh):
    python base_hotpotqa_distractors/check_baseline_match_outputs.py
    BASE_OUTDIR=/some/other/dir python .../check_baseline_match_outputs.py
"""
import os
import sys
from collections import Counter

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


def expected():
    for model in MODELS:
        for mode, arms in ARMS.items():
            for var in VARIANTS:
                for kind, a in arms:
                    path = os.path.join(BASE, mode, model, f"base_{var}_{kind}{a}.json")
                    yield model, mode, var, f"{kind}{a}", path


def main():
    rows = list(expected())
    present, missing = [], []
    for model, mode, var, arm, path in rows:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            present.append((model, mode, var, arm, os.path.getsize(path)))
        else:
            missing.append((model, mode, var, arm))

    print(f"results dir: {BASE}")
    print(f"expected arms: {len(rows)}   present: {len(present)}   missing: {len(missing)}")
    pc = Counter(m for m, *_ in present)
    for m in MODELS:
        tot = sum(1 for r in rows if r[0] == m)
        print(f"  {m:32s} {pc.get(m, 0):2d}/{tot} present")
    if missing:
        print("\nMISSING (unfinished or failed):")
        for model, mode, var, arm in missing:
            print(f"  {model}/{mode}/{var}_{arm}")
    else:
        print("\nAll expected arm outputs are present.")
    sys.exit(0 if not missing else 1)


if __name__ == "__main__":
    main()
