#!/bin/bash
# Quick smoke for the base-HotpotQA distractor sweep: small cohort, short servers, the 10-round
# hybrid variant, only fractions {0, 0.5}, throwaway output. Run on a LOGIN node:
#
#   bash base_hotpotqa_distractors/smoke.sh                 # qwen2.5-14b
#   ANSWER_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct ANSWER_SERVED=llama3.1-8b \
#     ANSWER_MAX_NUM_SEQS=128 bash base_hotpotqa_distractors/smoke.sh
#
# Verify: clean answers, an initial_distractor record per eligible question, and that
# sweep_summary shows gold_match at f=0.5 below f=0 (and distinct_answers above) — i.e. the
# distractors move the distribution. Then run launch.sh for the full sweep.
set -eo pipefail
exec env \
  OUTDIR="${OUTDIR:-$HOME/base_distractor_smoke}" \
  VARIANT="${VARIANT:-hybrid}" \
  MAX_Q="${MAX_Q:-20}" \
  NUM_RUNS="${NUM_RUNS:-2}" \
  FRACTIONS="${FRACTIONS:-0 0.5}" \
  SERVER_TIME="${SERVER_TIME:-01:30:00}" \
  CLIENT_TIME="${CLIENT_TIME:-01:00:00}" \
  bash "$(dirname "$0")/launch.sh"
