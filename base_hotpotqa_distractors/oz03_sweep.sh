#!/bin/bash
# oz03-hub-style HotpotQA collapse experiment with round-0 distractor seeding.
# Run on a LOGIN node:
#
#   bash base_hotpotqa_distractors/oz03_sweep.sh
#
# Setup (per the agreed design):
#   - ANSWER model: qwen2.5-14b (vLLM server) — also BUILDS the per-round AI docs from its own
#     answers (DOC_REUSE_ANSWER_SERVER=1), exactly like the oyilmazel/oz03-hub runs.
#   - ROUND-0 distractors: a SEPARATE strong model azure/gpt-5-mini via keymaker (DISTRACTOR_MODEL),
#     diverse_synth mode → several distinct, plausible wrong answers per question.
#   - 3 variants: search, hybrid (= replace-all: --num-synth-docs 10 --num-db-docs 0), replace_one.
#   - 10 questions, 3 rounds each, distractor fraction sweep {0, 0.3, 0.5, 0.7}.
#   - Per-run subsets OFF (shared context, oz03-hub style).
#
# Only the qwen answer server is started (no doc GPU server; distractors are api). API_KEY is read
# from .env at the repo root by run_sweep.sh. Outputs: per-variant sweep_summary_<variant>.json.
set -eo pipefail
exec env \
  DOC_REUSE_ANSWER_SERVER=1 \
  DISTRACTOR_MODEL="${DISTRACTOR_MODEL:-azure/gpt-5-mini}" \
  DISTRACTOR_MODEL_MODE="${DISTRACTOR_MODEL_MODE:-api}" \
  DISTRACTOR_MODE="${DISTRACTOR_MODE:-diverse_synth}" \
  VARIANTS="${VARIANTS:-search hybrid replace_one}" \
  NUM_ITERATIONS="${NUM_ITERATIONS:-3}" \
  MAX_Q="${MAX_Q:-10}" \
  NUM_RUNS="${NUM_RUNS:-10}" \
  FRACTIONS="${FRACTIONS:-0 0.3 0.5 0.7}" \
  OUTDIR="${OUTDIR:-$HOME/oz03_distractor_sweep}" \
  SERVER_TIME="${SERVER_TIME:-06:00:00}" \
  CLIENT_TIME="${CLIENT_TIME:-05:00:00}" \
  bash "$(dirname "$0")/launch.sh"
