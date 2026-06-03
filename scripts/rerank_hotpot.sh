#!/bin/bash
# --- SLURM ---
#SBATCH --job-name=rerank_hotpot
#SBATCH --output=logs/rerank_hotpot_%A.out
#SBATCH --error=logs/rerank_hotpot_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -C sm_70|sm_75|sm_80|sm_86|sm_90
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL

# Run from submit dir so paths resolve
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
fi

# --- Conda ---
module load cuda/12.6
module load conda/latest
conda activate rag

VLLM_API_BASE=http://gpu028.unity.rc.umass.edu:5151/v1
DOC_VLLM_API_BASE=http://gpu032.unity.rc.umass.edu:5150/v1

# Ensure a valid cache dir for HF
CACHE_DIR="/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache/"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"


if [[ -z "$VLLM_API_BASE" ]]; then
  echo "ERROR: VLLM_API_BASE is not set."
  exit 1
fi

if [[ -z "$DOC_VLLM_API_BASE" ]]; then
  echo "ERROR: DOC_VLLM_API_BASE is not set."
  exit 1
fi

echo "Using VLLM_API_BASE=$VLLM_API_BASE"
echo "Using DOC_VLLM_API_BASE=$DOC_VLLM_API_BASE"

# --- Config ---
MODEL="Qwen/Qwen2.5-14B-Instruct"
DOC_MODEL="Qwen/Qwen2.5-7B-Instruct"
OUTDIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/experiment_outputs/hotpotqa/$MODEL"

COMMON="--model-name $MODEL --doc-model-name $DOC_MODEL --num-runs 10 --chars-per-doc 500"

mkdir -p logs "$OUTDIR"

run() {
  python -u rerank_mitigation_hotpot_pipeline.py \
    --vllm-api-base "$VLLM_API_BASE" \
    --doc-vllm-api-base "$DOC_VLLM_API_BASE" \
    $COMMON "$@"
}

# --- Smoke test ---
# mkdir -p experiment_outputs/smoke_test
# run --lambda-penalty 0.1 --num-iterations 5 --max-questions 10 \
#   --output-path experiment_outputs/smoke_test/hotpot_rerank_lambda0.5.json

# --- Production runs ---
run --lambda-penalty 0.7 --max-questions 1400 --num-iterations 30 \
  --output-path "$OUTDIR/hotpot_rerank_lambda0.7.json"
