#!/bin/bash
# Quick smoke for the diverse_synth round-0 distractor workflow with a STRONG keymaker doc model.
# 1 round, 10 questions, qwen2.5-14b answerer, azure/gpt-5-mini doc generator, fractions {0, 0.7}.
# Run on a LOGIN node:
#
#   bash base_hotpotqa_distractors/smoke.sh                  # qwen2.5-14b + gpt-5-mini, diverse_synth
#
# Needs API_KEY in .env at the repo root (run_sweep.sh loads it). DOC_MODEL_MODE=api skips the doc
# GPU server entirely — only the qwen2.5-14b answer server is started.
#
# Verify in the output JSON / sweep_summary:
#   - an `initial_distractor` record per eligible question, with DISTINCT `injected_entity` values
#     and per-doc status `ok` | `fallback_template` (no refusal text, no gold leak),
#   - >=1 gold doc kept intact at fraction 0.7,
#   - round-0 `distinct_answers` at f=0.7 clearly ABOVE f=0 (the diversity lift).
# Then run launch.sh for the full multi-round sweep.
set -eo pipefail
exec env \
  OUTDIR="${OUTDIR:-$HOME/diverse_synth_smoke}" \
  VARIANT="${VARIANT:-search}" \
  NUM_ITERATIONS="${NUM_ITERATIONS:-1}" \
  MAX_Q="${MAX_Q:-10}" \
  NUM_RUNS="${NUM_RUNS:-5}" \
  FRACTIONS="${FRACTIONS:-0 0.7}" \
  DOC_MODEL_MODE="${DOC_MODEL_MODE:-api}" \
  DOC_MODEL="${DOC_MODEL:-azure/gpt-5-mini}" \
  DISTRACTOR_MODE="${DISTRACTOR_MODE:-diverse_synth}" \
  SERVER_TIME="${SERVER_TIME:-01:30:00}" \
  CLIENT_TIME="${CLIENT_TIME:-01:00:00}" \
  bash "$(dirname "$0")/launch.sh"
