"""
Eval-file analyses for the "Disproportionate Influence" section (Qwen-scoped).

Reads the per-round evaluation_outputs (small, ~7 MB) rather than the 400 MB raw
experiment files, and produces:

  1. citations-enabled audit  -> shows citation tracking is Qwen-only
  2. round-1 over-citation     -> obs/exp = AI cited share / AI available share
  3. collapse-before-saturation -> diversity (unique_words, ROUGE-L) declines while
                                   the pool is still mixed (beats 2 + 3 of the section)

UNITS NOTE: in the eval files, `ai_reference_percentage` is a FRACTION (0..1) but
`ai_citation_percentage` is a PERCENT (0..100). We normalize both to fractions
before dividing. (Forgetting this makes obs/exp look 100x too large.)

Run:  uv run --with numpy python eval_metrics.py
"""

import glob
import json
import os
import re

import numpy as np

# On Unity: /work/pi_dagarwal_umass_edu/project_4/file_storage/all_experiments/graphite/baseline.
# Override with $ALL_EXPERIMENTS_BASE.
BASE = os.environ.get("ALL_EXPERIMENTS_BASE",
                      "/work/pi_dagarwal_umass_edu/project_4/file_storage/all_experiments/graphite/baseline")
MODELS = {
    "Qwen": "Qwen/Qwen2.5-14B-Instruct",
    "DeepSeek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Llama": "meta-llama/Llama-3.1-8B-Instruct",
    "Mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}
VARIANTS = ["replace_one", "search", "replace_all"]


def citations_enabled_cheap(path):
    """Read citations_enabled from the metadata at the head of a big JSON without loading it."""
    with open(path) as f:
        head = f.read(4096)
    m = re.search(r'"citations_enabled"\s*:\s*(true|false)', head)
    return None if not m else (m.group(1) == "true")


def audit_citations():
    """Recursively find every raw experiment file (some are nested under owner/
    older_experiments subdirs) and report citations_enabled per model/variant."""
    print("=== citations_enabled audit (citation tracking is Qwen-only) ===")
    print(f"{'variant':<12} {'model':<10} {'citations_enabled':>18}  {'file'}")
    for v in VARIANTS:
        for mn, mp in MODELS.items():
            # recursive: matches both flat and nested (owner/, older_experiments/, etc.)
            paths = sorted(glob.glob(f"{BASE}/{v}/experiment_outputs/{mp}/**/local_{v}.json", recursive=True)
                           + glob.glob(f"{BASE}/{v}/experiment_outputs/{mp}/local_{v}.json"))
            paths = sorted(set(paths))
            if not paths:
                print(f"{v:<12} {mn:<10} {'(no raw file)':>18}")
            for p in paths:
                rel = p.split("experiment_outputs/")[1]
                print(f"{v:<12} {mn:<10} {str(citations_enabled_cheap(p)):>18}  {rel}")


def per_round(path, keys, NR=31):
    d = json.load(open(path))
    agg = {k: [[] for _ in range(NR)] for k in keys}
    for q in d["questions"]:
        for it in q["iterations"]:
            n = it["iteration_number"]
            if n >= NR:
                continue
            for k in keys:
                v = it["metrics"].get(k)
                if v is not None:
                    agg[k][n].append(v)
    return {k: [np.mean(x) if x else float("nan") for x in agg[k]] for k in keys}


def obs_exp(ref_frac, cite_pct):
    """ref is a fraction (0..1); cite is a percent (0..100). Return obs/exp."""
    rf = ref_frac if ref_frac <= 1.0 else ref_frac / 100.0
    cf = cite_pct / 100.0 if cite_pct > 1.0 else cite_pct
    return (cf / rf) if rf > 0 else float("nan"), rf, cf


def over_citation_round1():
    print("\n=== Round-1 over-citation (Qwen has citation data; others are 0/empty) ===")
    print(f"{'model':<10} {'variant':<12} {'ai_avail':>9} {'ai_cited':>9} {'obs/exp':>8}")
    for variant, fn in [("replace_one", "local_replace_one_eval.json"), ("search", "local_search_eval.json")]:
        for mn, mp in MODELS.items():
            p = f"{BASE}/{variant}/evaluation_outputs/{mp}/{fn}"
            if not os.path.exists(p):
                continue
            m = per_round(p, ["ai_reference_percentage", "ai_citation_percentage"])
            oe, rf, cf = obs_exp(m["ai_reference_percentage"][1], m["ai_citation_percentage"][1])
            print(f"{mn:<10} {variant:<12} {rf*100:>8.1f}% {cf*100:>8.1f}% {oe:>8.2f}")


def qwen_obs_exp_trajectory():
    print("\n=== Qwen Replace-One: per-round obs/exp + diversity (collapse before saturation) ===")
    p = f"{BASE}/replace_one/evaluation_outputs/{MODELS['Qwen']}/local_replace_one_eval.json"
    m = per_round(p, ["avg_pairwise_rougeL", "unique_words", "ai_reference_percentage", "ai_citation_percentage"], NR=20)
    print(f"{'rnd':>3} {'ai_avail':>9} {'obs/exp':>8} {'rougeL':>7} {'uniqWords':>9}")
    for n in range(20):
        oe, rf, _ = obs_exp(m["ai_reference_percentage"][n], m["ai_citation_percentage"][n])
        print(f"{n:>3} {rf*100:>8.1f}% {oe:>8.2f} {m['avg_pairwise_rougeL'][n]:>7.3f} {m['unique_words'][n]:>9.1f}")


def diversity_before_saturation():
    print("\n=== Beat 2: diversity declines during mixed window (multi-model), Replace-One ===")
    print(f"{'model':<10} {'uw_r1':>7} {'uw_r9':>7} {'rougeL_r1':>10} {'rougeL_r9':>10}  (ai_avail r9 ~95%)")
    for mn, mp in MODELS.items():
        p = f"{BASE}/replace_one/evaluation_outputs/{mp}/local_replace_one_eval.json"
        if not os.path.exists(p):
            continue
        m = per_round(p, ["unique_words", "avg_pairwise_rougeL"], NR=11)
        print(f"{mn:<10} {m['unique_words'][1]:>7.1f} {m['unique_words'][9]:>7.1f} "
              f"{m['avg_pairwise_rougeL'][1]:>10.3f} {m['avg_pairwise_rougeL'][9]:>10.3f}")


if __name__ == "__main__":
    audit_citations()
    over_citation_round1()
    qwen_obs_exp_trajectory()
    diversity_before_saturation()
