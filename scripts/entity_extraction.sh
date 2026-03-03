#!/bin/bash
#SBATCH --job-name=entity-extraction
#SBATCH --output=logs/entity-extraction_%A.out
#SBATCH --error=logs/entity-extraction_%A.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -C "vram40|vram48|vram80"
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

# Use a local HF cache so compute nodes don't need internet.
CACHE_DIR="$(pwd)/model_cache"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"

MODEL_SUBDIR="${MODEL_SUBDIR:-Qwen/Qwen2.5-14B-Instruct}"
# Modify the experiment directory as per the specific model and result path.
EXPERIMENT_DIR="${EXPERIMENT_DIR:-/work/pi_dagarwal_umass_edu/project_4/file_storage/oyilmazel_umass_edu/experiment_outputs/Qwen/Qwen2.5-14B-Instruct}"
OUT_DIR="${OUT_DIR:-entity_extraction_output/$MODEL_SUBDIR}"
MODEL_MODE="${MODEL_MODE:-local}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"

mkdir -p logs "$OUT_DIR"

python -u entity_extraction.py \
  --model-mode "$MODEL_MODE" \
  --model-name "$MODEL_NAME" \
  --gpu-mem-util "$GPU_MEM_UTIL" \
  --experiment-files "$EXPERIMENT_DIR/local_search.json" \
                     "$EXPERIMENT_DIR/local_replace_one.json" \
                     "$EXPERIMENT_DIR/local_replace_all.json" \
  --output-dir "$OUT_DIR"
