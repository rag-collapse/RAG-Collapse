#!/bin/bash
# Two-server native-distractor HotpotQA collapse experiment with EQUAL-WEIGHT diverse distractors.
# Run on a LOGIN node:
#
#   bash base_hotpotqa_distractors/equal_diverse_synth_sweep.sh
#
# Same setup as diverse_synth_sweep.sh (separate qwen2.5-14b answer + qwen2.5-7b-docgen servers, native
# 2-gold/8-distractor round-0 context, gold never corrupted, seed 42, gpt-5-mini distractors via the
# OpenAI API), with ONE change to how the 8 non-gold slots are filled:
#   - diverse_synth      : 8 DISTINCT wrong answers, one paragraph each.
#   - equal_diverse_synth: FEWER wrong answers ("topics"), each reinforced by DOCS_PER_TOPIC
#     paragraphs (default 2). 4 topics x 2 paragraphs = the 8 non-gold slots.
#
# Instead of a fraction sweep this sweeps the NUMBER OF TOPICS: T in {0,1,2,3,4} (T=0 = the matched
# no-distractor baseline). Outputs per-variant base_<variant>_t<T>.json + sweep_summary_<variant>.json.
#
# Needs OPENAI_API_KEY in .env at the repo root (run_sweep.sh loads it) for the gpt-5-mini distractors,
# and NATIVE_FILE pointing at the distractor-setting JSON.
SCR="${SCR:-/scratch4/workspace/oyilmazel_umass_edu-rag_collapse}"
exec env \
  DOC_MODEL_MODE="${DOC_MODEL_MODE:-server}" \
  DOC_MODEL="${DOC_MODEL:-qwen2.5-7b-docgen}" \
  DISTRACTOR_MODEL="${DISTRACTOR_MODEL:-gpt-5-mini}" \
  DISTRACTOR_MODEL_MODE="${DISTRACTOR_MODEL_MODE:-api}" \
  LITELLM_API_BASE="${LITELLM_API_BASE:-openai}" \
  DISTRACTOR_MODE="${DISTRACTOR_MODE:-equal_diverse_synth}" \
  DISTRACTOR_AVOID_GOLD="${DISTRACTOR_AVOID_GOLD:-1}" \
  INITIAL_DOCS="${INITIAL_DOCS:-native_distractor}" \
  NATIVE_FILE="${NATIVE_FILE:-$SCR/hotpot_dev_distractor_v1.json}" \
  VARIANTS="${VARIANTS:-search hybrid replace_one}" \
  NUM_ITERATIONS="${NUM_ITERATIONS:-3}" \
  TOPICS="${TOPICS:-0 1 2 3 4}" \
  DOCS_PER_TOPIC="${DOCS_PER_TOPIC:-2}" \
  MAX_Q="${MAX_Q:-10}" \
  NUM_RUNS="${NUM_RUNS:-10}" \
  SEED="${SEED:-42}" \
  OUTDIR="${OUTDIR:-$HOME/equal_distractor_sweep}" \
  SERVER_TIME="${SERVER_TIME:-06:00:00}" \
  CLIENT_TIME="${CLIENT_TIME:-05:00:00}" \
  bash "$(dirname "$0")/launch.sh"
