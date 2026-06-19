#!/bin/bash
# Misinfo experiment (3 variants x 3 arms) with Llama-3.1-8B-Instruct as the ANSWER model.
# Doc-gen is fixed Qwen2.5-7B. Run on a LOGIN node:
#   export GT_FILE=/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpot_dev_fullwiki_v1.json
#   bash scripts/hotpot-misinfo/launch_llama8b.sh
set -eo pipefail
cd "$(dirname "$0")/../.." || exit 1
BASE="${OUTPUT_BASE:-/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hotpotqa_distractor_experiment}"

export ANSWER_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
export ANSWER_SERVED="llama3.1-8b"
export ANSWER_EXTRA_ARGS=""
export OUTDIR="$BASE/$ANSWER_SERVED"

exec bash scripts/hotpot-misinfo/launch_all_variants.sh
