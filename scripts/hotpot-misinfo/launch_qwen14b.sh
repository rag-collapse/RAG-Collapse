#!/bin/bash
# Misinfo experiment (3 variants x 3 arms) with Qwen2.5-14B as the ANSWER model.
# Doc-gen is fixed Qwen2.5-7B. Run on a LOGIN node:
#   export GT_FILE=/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpot_dev_fullwiki_v1.json
#   bash scripts/hotpot-misinfo/launch_qwen14b.sh
set -eo pipefail
cd "$(dirname "$0")/../.." || exit 1
BASE="${OUTPUT_BASE:-/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hotpotqa_distractor_experiment}"

export ANSWER_MODEL_ID="Qwen/Qwen2.5-14B-Instruct"
export ANSWER_SERVED="qwen2.5-14b"
export ANSWER_EXTRA_ARGS=""
export ANSWER_PORT="${ANSWER_PORT:-5164}"   # distinct from baseline (5154) + other model launchers
export DOCGEN_PORT="${DOCGEN_PORT:-5163}"
export OUTDIR="$BASE/$ANSWER_SERVED"

exec bash scripts/hotpot-misinfo/launch_all_variants.sh
