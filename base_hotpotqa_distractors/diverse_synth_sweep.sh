#!/bin/bash
# Two-server native-distractor HotpotQA collapse experiment with round-0 distractor seeding.
# Run on a LOGIN node:
#
#   bash base_hotpotqa_distractors/diverse_synth_sweep.sh
#
# Setup (per the agreed design):
#   - SEPARATE servers: qwen2.5-14b ANSWER server + qwen2.5-7b-docgen DOC server. The doc server
#     builds the per-round AI docs from the answers (the two-server pattern).
#   - ROUND-0 source: the native HotpotQA distractor setting (2 gold + 8 distractors).
#   - ROUND-0 distractors: a strong model gpt-5-mini via the OpenAI API (DISTRACTOR_MODEL,
#     LITELLM_API_BASE=openai), diverse_synth mode, injected ONLY into the 8 non-gold slots
#     (--distractor-avoid-gold), randomly with seed 42. The 2 gold paragraphs are NEVER corrupted.
#   - 3 variants: search, hybrid (= replace-all), replace_one.
#   - 10 questions (set MAX_Q=150 for the full run), 3 rounds each, fraction sweep {0, 0.3, 0.5, 0.7}.
#   - Per-run subsets OFF (shared context).
#
# Needs OPENAI_API_KEY in .env at the repo root (run_sweep.sh loads it) for the gpt-5-mini distractors,
# and NATIVE_FILE pointing at the distractor-setting JSON. Outputs: per-variant sweep_summary_*.json.
SCR="${SCR:-/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu}"
exec env \
  DOC_MODEL_MODE="${DOC_MODEL_MODE:-server}" \
  DOC_MODEL="${DOC_MODEL:-qwen2.5-7b-docgen}" \
  DISTRACTOR_MODEL="${DISTRACTOR_MODEL:-gpt-5-mini}" \
  DISTRACTOR_MODEL_MODE="${DISTRACTOR_MODEL_MODE:-api}" \
  LITELLM_API_BASE="${LITELLM_API_BASE:-openai}" \
  DISTRACTOR_MODE="${DISTRACTOR_MODE:-diverse_synth}" \
  DISTRACTOR_AVOID_GOLD="${DISTRACTOR_AVOID_GOLD:-1}" \
  INITIAL_DOCS="${INITIAL_DOCS:-native_distractor}" \
  NATIVE_FILE="${NATIVE_FILE:-$SCR/hotpot_dev_distractor_v1.json}" \
  VARIANTS="${VARIANTS:-search hybrid replace_one}" \
  NUM_ITERATIONS="${NUM_ITERATIONS:-3}" \
  MAX_Q="${MAX_Q:-10}" \
  NUM_RUNS="${NUM_RUNS:-10}" \
  FRACTIONS="${FRACTIONS:-0 0.3 0.5 0.7}" \
  SEED="${SEED:-42}" \
  OUTDIR="${OUTDIR:-$HOME/distractor_sweep}" \
  SERVER_TIME="${SERVER_TIME:-06:00:00}" \
  CLIENT_TIME="${CLIENT_TIME:-05:00:00}" \
  bash "$(dirname "$0")/launch.sh"
