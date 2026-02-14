#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=logs/pipeline_%A.out
#SBATCH --error=logs/pipeline_%A.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -C "vram40|vram48&sm_70|sm_75|sm_80|sm_86|sm_89|sm_90"
#SBATCH --cpus-per-task=2
# vLLM: use Triton instead of FlashInfer (FlashInfer JIT needs nvcc/CUDA_HOME)
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0

# FlashInfer may still be imported and requires CUDA_HOME/nvcc. Load CUDA module (Unity: cuda/11.8, 12.1, 12.4.1, 12.6, 12.8, 13.1).
if command -v module &>/dev/null; then
  module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
fi
if [ -z "${CUDA_HOME}" ] && command -v nvcc &>/dev/null; then
  export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
fi

# (optional but recommended on Unity) keep caches off $HOME
export HF_HOME=/work/pi_dagarwal_umass_edu/project_4/ffatima/.cache/hf
export VLLM_CACHE_ROOT=/work/pi_dagarwal_umass_edu/project_4/ffatima/.cache/vllm


module load conda/latest
conda activate ragenv

DATASET="datasets/umass_data.entity.chatgpt.50.jsonl"
OUTDIR="experiment_outputs"
mkdir -p "$OUTDIR"

# Shared args (adjust max-questions, num-runs, chars-per-doc as needed)
COMMON="--dataset-path $DATASET --num-runs 10 --chars-per-doc 400"
# Optional: cap rounds for quicker runs (e.g. 2 instead of 10/20/30)
# EXTRA="--max-iterations 2 --max-questions 2"
EXTRA="--max-questions 8"

# =============================================================================
# API mode (uncomment to use; set API_KEY for completion and for search embeddings)
# =============================================================================
# python -u pipeline.py --model-mode api --model-name openai/gpt4o \
#   --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 $COMMON $EXTRA \
#   --output-path "$OUTDIR/api_hybrid_replace_all_style.json"
#
# python -u pipeline.py --model-mode api --model-name openai/gpt4o \
#   --pipeline-variant hybrid --num-synth-docs 1 --num-db-docs 3 $COMMON $EXTRA \
#   --output-path "$OUTDIR/api_hybrid_fixed_mix.json"
#
# python -u pipeline.py --model-mode api --model-name openai/gpt4o \
#   --pipeline-variant search $COMMON $EXTRA \
#   --output-path "$OUTDIR/api_search.json"

# =============================================================================
# Local mode (uncomment to use; requires GPU)
# =============================================================================
# Hybrid: one pipeline; ratio controlled by --num-synth-docs / --num-db-docs
# replace_all-style: all synthetic, no refs
python -u pipeline.py --model-mode local --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 $COMMON $EXTRA \
  --output-path "$OUTDIR/local_hybrid_replace_all_style.json"

# fixed mix: 1 synthetic + 3 original refs (default)
python -u pipeline.py --model-mode local --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --pipeline-variant hybrid --num-synth-docs 1 --num-db-docs 3 $COMMON $EXTRA \
  --output-path "$OUTDIR/local_hybrid_fixed_mix.json"

# Search variant uses LiteLLM embeddings (API_KEY required; same api_base as completion).
python -u pipeline.py --model-mode local --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --pipeline-variant search $COMMON $EXTRA \
  --output-path "$OUTDIR/local_search.json"
