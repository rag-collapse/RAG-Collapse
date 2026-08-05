#!/bin/bash
# Entity extraction for the CROSS-MODEL baseline runs — computes entity-collapse metrics
# (unique entities, entity similarity, per-run canonical_entities) from the MAIN model's answers
# (run["answer"]) of the cross-model experiment JSONs, so the entity view matches the actual
# runs (not the separately-tagged workshop dump). Writes <variant>_entity_results.json next to
# each run under .../<side>/<variant>/entity_extraction_output/.
#
# Usage (on a login node, submit):
#   SIDE=deepseek-r1-distill-qwen-7b VARIANTS="replace_all replace_one" sbatch --export=ALL scripts/cross_model_baseline/entity_cross_model.sh
#
#SBATCH --job-name=xm-entity
#SBATCH --output=logs/xm_entity_%A.out
#SBATCH --error=logs/xm_entity_%A.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=bf16
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR" || exit 1; fi

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
module load conda/latest
module load cuda/12.6
conda activate ragenv
export VLLM_USE_FLASHINFER=0          # avoid a FlashInfer init error on some nodes
export VLLM_USE_DEEP_GEMM=0           # avoid the DeepGEMM FP8 warmup abort on Hopper GPUs
CACHE_DIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hf_cache"
mkdir -p "$CACHE_DIR"; export HF_HOME="$CACHE_DIR"; export HF_HUB_CACHE="$CACHE_DIR"

ENTITY_MODEL="${ENTITY_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"   # light extraction model
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"
XB="${XB:-/work/pi_dagarwal_umass_edu/project_4/file_storage/all_experiments/graphite/cross-model-baseline/Qwen/Qwen2.5-14B-Instruct}"
SIDE="${SIDE:?set SIDE (side model served name, e.g. deepseek-r1-distill-qwen-7b)}"
VARIANTS="${VARIANTS:-replace_all replace_one}"
mkdir -p logs

for v in $VARIANTS; do
  in="$XB/$SIDE/$v/experiment_outputs/local_${v}.json"
  out="$XB/$SIDE/$v/entity_extraction_output"
  if [[ ! -f "$in" ]]; then echo "SKIP (missing): $in"; continue; fi
  mkdir -p "$out"
  echo "=== entity extraction: $SIDE/$v -> $out ==="
  python -u entity_extraction.py \
    --model-mode local --model-name "$ENTITY_MODEL" --gpu-mem-util "$GPU_MEM_UTIL" \
    --experiment-files "$in" --output-dir "$out"
done
echo "=== done: $SIDE ($VARIANTS) ==="
