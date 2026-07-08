#!/bin/bash
# Text-similarity evaluation for the CROSS-MODEL baseline runs.
#
# Runs evaluation.py (ROUGE / TES / cosine similarity, unique_words, ai_reference%,
# same_answer judge) over the MAIN model's answers (run["answer"]) for each variant.
# evaluation.py has NO entity dependency, so this is the complete non-entity eval.
# We deliberately do NOT run entity_extraction.py here (unique_entities / entity_similarity
# / collapse-by-simulation are entity-based and intentionally skipped).
#
# Reads:  <ALL_EXP_BASE>/graphite/cross-model-baseline/<variant>/experiment_outputs/<MODEL_SUBDIR>/local_<variant>.json
# Writes: <ALL_EXP_BASE>/graphite/cross-model-baseline/<variant>/evaluation_outputs/<MODEL_SUBDIR>/local_<variant>_eval.json
#
# --- SLURM ---
#SBATCH --job-name=xmodel-eval
#SBATCH --output=logs/xmodel_eval_%A.out
#SBATCH --error=logs/xmodel_eval_%A.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=bf16
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL

set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR" || exit 1; fi

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"

module load conda/latest
conda activate ragenv
module load cuda/12.6
nvidia-smi || true

# Shared HF cache on scratch (same one the cross-model run used).
CACHE_DIR="${CACHE_DIR:-/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache}"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"

# Same-answer judge model (local vLLM, needs the GPU) — match the baseline eval default.
export SAME_ANSWER_MODEL_NAME="${SAME_ANSWER_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"

ALL_EXP_BASE="${ALL_EXP_BASE:-/work/pi_dagarwal_umass_edu/project_4/file_storage/all_experiments}"
XMODEL_BASE="$ALL_EXP_BASE/graphite/cross-model-baseline"
MODEL_SUBDIR="${MODEL_SUBDIR:-Qwen/Qwen2.5-14B-Instruct}"   # the MAIN (measured) model
SIDE="${SIDE:?set SIDE (side model served name, e.g. deepseek-r1-distill-qwen-7b / llama-3.1-8b / mistral-7b)}"

mkdir -p logs

# Variants to evaluate; missing ones (e.g. search if it timed out) are skipped gracefully.
VARIANTS="${VARIANTS:-replace_all replace_one search}"

# Layout: cross-model-baseline/<main>/<side>/<variant>/{experiment_outputs,evaluation_outputs}/
for variant in $VARIANTS; do
  base="$XMODEL_BASE/$MODEL_SUBDIR/$SIDE/$variant"
  in_file="$base/experiment_outputs/local_${variant}.json"
  out_dir="$base/evaluation_outputs"
  out_file="$out_dir/local_${variant}_eval.json"
  if [[ -f "$in_file" ]]; then
    mkdir -p "$out_dir"
    echo "=== eval $variant : $in_file -> $out_file ==="
    python -u evaluation.py "$in_file" "$out_file" --cache-dir "$CACHE_DIR"
  else
    echo "Skipping missing experiment file: $in_file"
  fi
done

echo "=== cross-model eval done ==="
