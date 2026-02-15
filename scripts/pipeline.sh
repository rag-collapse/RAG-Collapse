#!/bin/bash
# Run RAG collapse pipeline (hybrid + search) on SLURM. Submit with: sbatch scripts/pipeline.sh

# --- SLURM ---
#SBATCH --job-name=pipeline
#SBATCH --output=logs/pipeline_%A.out
#SBATCH --error=logs/pipeline_%A.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -C "vram40|vram48&sm_70|sm_75|sm_80|sm_86|sm_89|sm_90"
#SBATCH --cpus-per-task=2

# --- Environment (vLLM, CUDA, caches) ---
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
if command -v module &>/dev/null; then
  module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
fi
[ -z "${CUDA_HOME}" ] && command -v nvcc &>/dev/null && export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")

export HF_HOME="${HF_HOME:-/work/pi_dagarwal_umass_edu/project_4/ffatima/.cache/hf}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/work/pi_dagarwal_umass_edu/project_4/ffatima/.cache/vllm}"

# --- Conda ---
module load conda/latest
conda activate ragenv

# --- Config (edit as needed) ---
DATASET="datasets/umass_data.entity.chatgpt.50.jsonl"
OUTDIR="experiment_outputs"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
COMMON="--dataset-path $DATASET --num-runs 10 --chars-per-doc 400"
EXTRA="--max-questions 8"

mkdir -p logs "$OUTDIR"

# --- Local mode (GPU): two hybrid configs + search ---
run_local() {
  python -u pipeline.py --model-mode local --model-name "$MODEL" $COMMON $EXTRA "$@"
}

run_local --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 \
  --output-path "$OUTDIR/local_hybrid_replace_all.json"

run_local --pipeline-variant hybrid --num-synth-docs 1 --num-db-docs 3 \
  --output-path "$OUTDIR/local_hybrid_replace_one.json"

run_local --pipeline-variant search --output-path "$OUTDIR/local_search.json"

# --- API mode (uncomment and set API_KEY; comment out Local block above) ---
# MODEL_API="openai/gpt4o"
# run_api() { python -u pipeline.py --model-mode api --model-name "$MODEL_API" $COMMON $EXTRA "$@"; }
# run_api --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 --output-path "$OUTDIR/api_hybrid_replace_all.json"
# run_api --pipeline-variant hybrid --num-synth-docs 1 --num-db-docs 3 --output-path "$OUTDIR/api_hybrid_replace_one.json"
# run_api --pipeline-variant search --output-path "$OUTDIR/api_search.json"
