#!/bin/bash
# Two-server native-distractor HotpotQA collapse experiment, diverse_synth distractors,
# with PER-RUN CONTEXT SHUFFLING. Run on a LOGIN node:
#
#   bash base_hotpotqa_distractors/shuffled_diverse_synth_sweep.sh
#
# Identical to diverse_synth_sweep.sh (separate qwen2.5-14b answer + qwen2.5-7b-docgen servers,
# native 2-gold/8-distractor round-0 context, gold never corrupted, seed 42, gpt-5-mini distractors
# via the OpenAI API, fraction sweep {0, 0.3, 0.5, 0.7}, 3 variants, 3 rounds) with ONE change:
#   - SHUFFLE_PER_RUN=1 -> each of the NUM_RUNS parallel generations sees the SAME context docs
#     but in a DIFFERENT, seeded random ORDER (re-shuffled every round). This isolates how much of
#     the answer concentration is a fixed-doc-order artifact vs. genuine model commitment.
#
# Order randomness is reproducible from SEED (string-seeded random.Random per question/round/run,
# PYTHONHASHSEED-independent). Answer sampling is set to temperature=1.0 / top_p=1.0 to maximize the
# spread of the 10 parallel generations (the doc model inherits this temperature). Uses its OWN server
# ports (5182/5183) so it can run IN PARALLEL with diverse_synth/equal_diverse_synth on 5180/5181.
# Needs OPENAI_API_KEY in .env at the repo root.
SCR="${SCR:-/scratch4/workspace/oyilmazel_umass_edu-rag_collapse}"
exec env \
  ANSWER_PORT="${ANSWER_PORT:-5182}" \
  DOCGEN_PORT="${DOCGEN_PORT:-5183}" \
  TEMPERATURE="${TEMPERATURE:-1.0}" \
  TOP_P="${TOP_P:-1.0}" \
  DOC_MODEL_MODE="${DOC_MODEL_MODE:-server}" \
  DOC_MODEL="${DOC_MODEL:-qwen2.5-7b-docgen}" \
  DISTRACTOR_MODEL="${DISTRACTOR_MODEL:-gpt-5-mini}" \
  DISTRACTOR_MODEL_MODE="${DISTRACTOR_MODEL_MODE:-api}" \
  LITELLM_API_BASE="${LITELLM_API_BASE:-openai}" \
  DISTRACTOR_MODE="${DISTRACTOR_MODE:-diverse_synth}" \
  DISTRACTOR_AVOID_GOLD="${DISTRACTOR_AVOID_GOLD:-1}" \
  SHUFFLE_PER_RUN="${SHUFFLE_PER_RUN:-1}" \
  INITIAL_DOCS="${INITIAL_DOCS:-native_distractor}" \
  NATIVE_FILE="${NATIVE_FILE:-$SCR/hotpot_dev_distractor_v1.json}" \
  VARIANTS="${VARIANTS:-search hybrid replace_one}" \
  NUM_ITERATIONS="${NUM_ITERATIONS:-3}" \
  MAX_Q="${MAX_Q:-10}" \
  NUM_RUNS="${NUM_RUNS:-10}" \
  FRACTIONS="${FRACTIONS:-0 0.3 0.5 0.7}" \
  SEED="${SEED:-42}" \
  OUTDIR="${OUTDIR:-$HOME/shuffled_distractor_sweep}" \
  SERVER_TIME="${SERVER_TIME:-06:00:00}" \
  CLIENT_TIME="${CLIENT_TIME:-05:00:00}" \
  bash "$(dirname "$0")/launch.sh"
