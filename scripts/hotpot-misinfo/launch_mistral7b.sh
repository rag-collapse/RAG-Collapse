#!/bin/bash
# Misinfo experiment (3 variants x 3 arms) with Mistral-7B-Instruct-v0.3 as the ANSWER model.
# Doc-gen is fixed Qwen2.5-7B. Run on a LOGIN node:
#   export GT_FILE=/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpot_dev_fullwiki_v1.json
#   bash scripts/hotpot-misinfo/launch_mistral7b.sh
set -eo pipefail
cd "$(dirname "$0")/../.." || exit 1
BASE="${OUTPUT_BASE:-/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hotpotqa_distractor_experiment}"

export ANSWER_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
export ANSWER_SERVED="mistral-7b"
export ANSWER_EXTRA_ARGS="--tokenizer-mode mistral"   # Mistral tokenizer needs this (no HF chat_template)
export OUTDIR="$BASE/$ANSWER_SERVED"

exec bash scripts/hotpot-misinfo/launch_all_variants.sh
