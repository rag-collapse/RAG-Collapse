#!/bin/bash
# --- SLURM ---
#SBATCH --job-name=evaluation
#SBATCH --output=logs/evaluation_%A.out
#SBATCH --error=logs/evaluation_%A.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -C "vram48|vram80"
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# Run from submit dir so paths resolve
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
fi

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"

# --- Conda ---
module load conda/latest
conda activate ragenv

module load cuda/12.6

nvidia-smi

# Use a local HF cache so compute nodes don't need internet.
CACHE_DIR="/work/pi_dagarwal_umass_edu/hf_cache/"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"
# Evaluate multiple Qwen models in one job.
# Uncomment the model you want to evaluate.
MODEL_SUBDIRS=(
  #"Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  # "Qwen/Qwen2.5-1.5B-Instruct"  
)

mkdir -p logs

for MODEL_SUBDIR in "${MODEL_SUBDIRS[@]}"; do
  EXPERIMENT_DIR="experiment_outputs/$MODEL_SUBDIR"
  OUT_DIR="evaluation_outputs/$MODEL_SUBDIR"
  mkdir -p "$OUT_DIR"

  # Use the current generator model as the same-answer judge model by default.
  export SAME_ANSWER_MODEL_NAME="$MODEL_SUBDIR"

  for base in local_search local_replace_one local_replace_all; do
    in_file="$EXPERIMENT_DIR/${base}.json"
    out_file="$OUT_DIR/${base}_eval.json"
    if [[ -f "$in_file" ]]; then
      python -u evaluation.py "$in_file" "$out_file" --cache-dir "$CACHE_DIR"
    else
      echo "Skipping missing experiment file: $in_file"
    fi
  done
done
