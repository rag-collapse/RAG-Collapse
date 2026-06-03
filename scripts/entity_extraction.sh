#!/bin/bash
#SBATCH --job-name=entity-mistral-7bv0.3-agentic-rag
#SBATCH --output=logs/entity-extraction_%A.out
#SBATCH --error=logs/entity-extraction_%A.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -C "vram48|vram80"
#SBATCH --cpus-per-task=2

set -eo pipefail

# Run from submit dir so paths resolve
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
fi

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"

module load conda/latest
module load cuda/12.6
# activate conda environment
conda activate ragenv
# Used to resolve the FlashInference error.
export VLLM_USE_FLASHINFER=0

CACHE_DIR="/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"

# ── Main model (change only this line to switch pipelines) ────────────────────
MAIN_MODEL="${MAIN_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

# ── Derived paths ─────────────────────────────────────────────────────────────
EXPERIMENT_DIR="${EXPERIMENT_DIR:-/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/experiment_outputs/$MAIN_MODEL}"
OUT_DIR="${OUT_DIR:-entity_extraction_output/$MAIN_MODEL}"

# ── Entity extraction model (lighter model used for extraction only) ──────────
ENTITY_MODEL="${ENTITY_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_MODE="${MODEL_MODE:-local}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"

mkdir -p logs "$OUT_DIR"

python -u entity_extraction.py \
  --model-mode "$MODEL_MODE" \
  --model-name "$ENTITY_MODEL" \
  --gpu-mem-util "$GPU_MEM_UTIL" \
  --experiment-files "$EXPERIMENT_DIR/local_agentic_rag.json" \
  --output-dir "$OUT_DIR"
