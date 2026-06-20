#!/bin/bash
# Quick smoke for the original-HotpotQA distractor-setting experiment: hybrid (10 rounds),
# 20 questions, num_runs 2, both arms (distractor + gold-only), short servers, throwaway output.
# Run on a LOGIN node:
#   bash base_hotpotqa_distractors/smoke_native.sh
# Verify: clean answers, distractor-setting gold_match <= gold-only gold_match, runs end-to-end.
set -eo pipefail
exec env \
  OUTDIR="${OUTDIR:-$HOME/base_native_smoke}" \
  VARIANT="${VARIANT:-hybrid}" \
  MAX_Q="${MAX_Q:-20}" \
  NUM_RUNS="${NUM_RUNS:-2}" \
  ARMS="${ARMS:-distractor goldonly}" \
  SERVER_TIME="${SERVER_TIME:-01:30:00}" \
  CLIENT_TIME="${CLIENT_TIME:-01:00:00}" \
  bash "$(dirname "$0")/launch_native.sh"
