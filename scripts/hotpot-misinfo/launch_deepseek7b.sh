#!/bin/bash
# Misinfo experiment (3 variants x 3 arms) with DeepSeek-R1-Distill-Qwen-7B as the ANSWER model.
# Doc-gen is fixed Qwen2.5-7B. Run on a LOGIN node:
#   (GT_FILE defaults to the native HotpotQA JSON; override via env only if it moves)
#   bash scripts/hotpot-misinfo/launch_deepseek7b.sh
#
# DeepSeek-R1 is a reasoning model:
#   NO --reasoning-parser: on vLLM 0.20 the detokenizer returns this model's content as
#   byte-level BPE (Ġ/Ċ) REGARDLESS of the parser; server_llm.py (_clean_response) byte-decodes
#   it and strips any <think> block client-side.
#   MAX_TOKENS=4096 gives the trace room (512 would truncate it and can leave content empty).
set -eo pipefail
cd "$(dirname "$0")/../.." || exit 1
BASE="${OUTPUT_BASE:-/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hotpotqa_distractor_experiment}"

export ANSWER_MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
export ANSWER_SERVED="deepseek-r1-distill-qwen-7b"
export ANSWER_EXTRA_ARGS=""                 # no reasoning parser (see note above); <think> stripped in server_llm.py
export ANSWER_MAX_NUM_SEQS=128              # 7B model -> matches start_llm_server_deepseek_r1_distill_qwen7b.sh
export MAX_TOKENS="${MAX_TOKENS:-4096}"     # answer headroom for the reasoning trace; docs stay 512
export ANSWER_PORT="${ANSWER_PORT:-5170}"   # distinct from baseline (5154) + other model launchers
export DOCGEN_PORT="${DOCGEN_PORT:-5169}"
export OUTDIR="$BASE/$ANSWER_SERVED"

exec bash scripts/hotpot-misinfo/launch_all_variants.sh
