#!/bin/bash
# --- SLURM ---
#SBATCH --job-name=evaluation-mistral-7bv0.3-agentic-rag
#SBATCH --output=logs/evaluation_%A.out
#SBATCH --error=logs/evaluation_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -C "vram48|vram80"
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=EBGIN,END,FAIL

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
CACHE_DIR="/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache/"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"

# --- Config (edit as needed) ---
# Model subdir used for evaluation outputs and judge model name, e.g. Qwen/Qwen2.5-14B-Instruct.
MODEL_SUBDIR="${MODEL_SUBDIR:-mistralai/Mistral-7B-Instruct-v0.3}"
# Input/output on shared file storage (experiment_outputs read from here, evaluation_outputs written here).
INDIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/experiment_outputs/$MODEL_SUBDIR"
OUT_DIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/evaluation_outputs/$MODEL_SUBDIR"
mkdir -p logs "$OUT_DIR"

EXPERIMENT_DIR="$INDIR"

export SAME_ANSWER_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

for base in local_search local_replace_one local_replace_all local_agentic_rag; do
  in_file="$EXPERIMENT_DIR/${base}.json"
  out_file="$OUT_DIR/${base}_eval.json"
  if [[ -f "$in_file" ]]; then
    python -u evaluation.py "$in_file" "$out_file" --cache-dir "$CACHE_DIR"
  else
    echo "Skipping missing experiment file: $in_file"
  fi
done
