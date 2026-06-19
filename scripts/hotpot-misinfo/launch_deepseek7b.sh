#!/bin/bash
# Misinfo experiment (3 variants x 3 arms) with DeepSeek-R1-Distill-Qwen-7B as the ANSWER model.
# Doc-gen is fixed Qwen2.5-7B. Run on a LOGIN node:
#   export GT_FILE=/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpot_dev_fullwiki_v1.json
#   bash scripts/hotpot-misinfo/launch_deepseek7b.sh
#
# DeepSeek-R1 is a reasoning model:
#   --reasoning-parser deepseek_r1 keeps the <think> trace OUT of message.content (else F1 is junk).
#   MAX_TOKENS=4096 gives the trace room (512 would truncate it and can leave content empty).
set -eo pipefail
cd "$(dirname "$0")/../.." || exit 1
BASE="${OUTPUT_BASE:-/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hotpotqa_distractor_experiment}"

export ANSWER_MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
export ANSWER_SERVED="deepseek-r1-distill-qwen-7b"
export ANSWER_EXTRA_ARGS="--reasoning-parser deepseek_r1"
export ANSWER_MAX_NUM_SEQS=128              # 7B model -> matches start_llm_server_deepseek_r1_distill_qwen7b.sh
export MAX_TOKENS="${MAX_TOKENS:-4096}"     # answer headroom for the reasoning trace; docs stay 512
export OUTDIR="$BASE/$ANSWER_SERVED"

exec bash scripts/hotpot-misinfo/launch_all_variants.sh
