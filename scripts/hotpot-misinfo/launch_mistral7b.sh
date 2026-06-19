#!/bin/bash
# Misinfo experiment (3 variants x 3 arms) with Mistral-7B-Instruct-v0.3 as the ANSWER model.
# Doc-gen is fixed Qwen2.5-7B. Run on a LOGIN node:
#   (GT_FILE defaults to the native HotpotQA JSON; override via env only if it moves)
#   bash scripts/hotpot-misinfo/launch_mistral7b.sh
set -eo pipefail
cd "$(dirname "$0")/../.." || exit 1
BASE="${OUTPUT_BASE:-/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hotpotqa_distractor_experiment}"

export ANSWER_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
export ANSWER_SERVED="mistral-7b"
export ANSWER_EXTRA_ARGS="--tokenizer-mode mistral"   # Mistral tokenizer needs this (no HF chat_template)
export ANSWER_MAX_NUM_SEQS=128                        # 7B model -> matches start_llm_server_mistral7b.sh
export ANSWER_PORT="${ANSWER_PORT:-5166}"   # distinct from baseline (5154) + other model launchers
export DOCGEN_PORT="${DOCGEN_PORT:-5165}"
export OUTDIR="$BASE/$ANSWER_SERVED"

exec bash scripts/hotpot-misinfo/launch_all_variants.sh
