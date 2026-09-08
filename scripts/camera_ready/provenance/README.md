# §6.3 / §6.5 provenance analysis (self-preference in citations)

This is the reproducibility code for the paper's "why does collapse happen" result. It shows that
during recursive RAG the model cites its own prior generations far above their availability, and that
the over-citation is not explained by AI-authorship or by reference quality. That is the self-preference
finding in §6.3 and §6.5.

Authored by Greg Druck. Vendored here from the `collapse-randomness-research` repo
(`model_collapse/workshop_paper/`) so this repo reproduces the numbers on its own. The citation mechanism
is computed for Qwen2.5-14B only, because it is the one model whose baseline run has `citations_enabled`.
The diversity collapse itself is multi-model and lives elsewhere in the paper.

## Pipeline

Run order for the quality-controlled regression (§6.5):

1. `build_qwen_scoring_input.py` extracts the Qwen Replace-One round-1 references into
   `qwen_round1_to_score.jsonl`. Point it at the raw run with argv or `$QWEN_REPLACE_ONE_RUN`.
2. `score_references_qwen.py` scores each reference on the 8 quality dimensions with an LLM judge, writing
   `qwen_round1_scored.jsonl`. Needs `ANTHROPIC_API_KEY`.
3. `provenance_regression.py` reads the scored file plus the detector labels and reports citation rate by
   provenance group, the pairwise contrasts, and the OLS of citation rate on provenance controlling for the
   8 quality dimensions.

Corroboration and the other §6 citation numbers:

- `provenance_full.py <run.json>` runs the full 400-question three-way citation-rate analysis with proxy
  controls (length, position, lexical relevance), since the LLM scores cover only the round-1 subset.
- `eval_metrics.py` reads the small `evaluation_outputs` and produces the citations-enabled audit, the
  round-1 obs/exp over-citation, and the collapse-before-saturation check. Point it at the baseline tree
  with `$ALL_EXPERIMENTS_BASE`.
- `search_twoway.py <run.json>` is the Search-setting version (self vs originals-combined, since Search
  chunks carry no URL for the AI/human split).

## Inputs

- **Detector labels.** `ai_detector.csv` is vendored here (GPTZero output, batch `January_2026`). The
  scripts resolve it relative to this directory.
- **Raw Qwen run.** The Qwen2.5-14B Replace-One `local_replace_one.json`. On Unity it is
  `/work/pi_dagarwal_umass_edu/project_4/file_storage/all_experiments/graphite/baseline/replace_one/experiment_outputs/Qwen/Qwen2.5-14B-Instruct/local_replace_one.json`.
  It is ~400 MB, so it is not vendored. Pass it as argv or via the env vars above.
- **Scored references.** `qwen_round1_scored.jsonl` is the one intermediate not vendored, because it is an
  LLM-judge output. Regenerate it with steps 1 and 2 (needs `ANTHROPIC_API_KEY`), or drop in a saved copy.
  Only `provenance_regression.py` needs it. `provenance_full.py` and `eval_metrics.py` do not.

## Key results this reproduces (Qwen2.5-14B, Replace-One round 1, 378 q / 3,296 refs)

- Citation rate by provenance. Self-generated **0.281**, AI-written **0.123**, human-written **0.078**.
- Self minus human is **+0.203**. AI-written minus human is **+0.045** (p<0.001), real but far smaller.
  Over-citation tracks self-generation, not AI authorship.
- OLS controlling for all 8 quality dimensions. `is_answer_derived` = **+0.154** (95% CI [+0.115, +0.191],
  p<0.001). `is_ai_original` = +0.004 (p=0.65, not significant). Self-generation survives quality control,
  AI authorship does not.
- Round-1 over-citation from the official eval metric. Replace-One obs/exp **2.21×**, Search **1.62×**.

## Portability note

The detector reads are pinned to `encoding="utf-8"` so the CSV parses on Windows as well as Linux. The
analysis is otherwise numpy and standard library only.
