#!/bin/bash
# --- SLURM ---
#SBATCH --job-name=rerank_pipeline
#SBATCH --output=logs/rerank_pipeline_%A.out
#SBATCH --error=logs/rerank_pipeline_%A.err
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
CACHE_DIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hf_cache/"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"

if [[ -z "$VLLM_API_BASE" ]]; then
  echo "ERROR: VLLM_API_BASE is not set. Start the vLLM server first, then:"
  echo "  export VLLM_API_BASE=\"http://<fqdn>:5150/v1\""
  echo "  sbatch --export=ALL scripts/rerank_pipeline.sh"
  exit 1
fi

if [[ -z "$DOC_VLLM_API_BASE" ]]; then
  echo "ERROR: DOC_VLLM_API_BASE is not set."
  exit 1
fi

echo "Using VLLM_API_BASE=$VLLM_API_BASE"
echo "Using DOC_VLLM_API_BASE=$DOC_VLLM_API_BASE"

# --- Config ---
DATASET="datasets/umass_data.entity.chatgpt.400.jsonl"
MODEL="Qwen/Qwen2.5-14B-Instruct"
DOC_MODEL="Qwen/Qwen2.5-7B-Instruct"
OUTDIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/experiment_outputs/$MODEL"

COMMON="--dataset-path $DATASET --model-name $MODEL --doc-model-name $DOC_MODEL --chars-per-doc 400 --num-runs 10"
EXTRA="--max-questions 400"

mkdir -p logs "$OUTDIR"

run() {
  python -u rerank_mitigation_pipeline.py \
    --model-mode server --vllm-api-base "$VLLM_API_BASE" \
    --doc-model-mode server --doc-vllm-api-base "$DOC_VLLM_API_BASE" \
    $COMMON $EXTRA "$@"
}

# --- Smoke test ---
# mkdir -p experiment_outputs/smoke_test
# run --lambda-penalty 0.5 --num-iterations 5 --max-questions 10 \
#   --output-path experiment_outputs/smoke_test/rerank_lambda0.5.json

# --- Production runs ---
# Uncomment the lambda values you want to run:
# run --lambda-penalty 0.1 --num-iterations 30 --output-path "$OUTDIR/rerank_lambda0.1.json"
run --lambda-penalty 0.5 --num-iterations 30 --output-path "$OUTDIR/rerank_lambda0.5.json"
# run --lambda-penalty 0.7 --num-iterations 30 --output-path "$OUTDIR/rerank_lambda0.7.json"
#run --lambda-penalty 1.0 --num-iterations 30 --output-path "$OUTDIR/rerank_lambda1.0.json"
