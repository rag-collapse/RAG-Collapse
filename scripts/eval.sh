#!/bin/bash
# --- SLURM ---
#SBATCH --job-name=evaluation
#SBATCH --output=logs/evaluation_%A.out
#SBATCH --error=logs/evaluation_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram48|vram80
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
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
CACHE_DIR="/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache_oz/"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"

# Model subdir used for evaluation outputs and judge model name, e.g. Qwen/Qwen2.5-14B-Instruct.
MODEL_SUBDIR="${MODEL_SUBDIR:-Qwen/Qwen2.5-14B-Instruct}"
# Input/output on shared file storage (experiment_outputs read from here, evaluation_outputs written here).

# To evaluate hotpot stuff, use the below
INDIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/experiment_outputs/hotpotqa/$MODEL_SUBDIR"
OUT_DIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/evaluation_outputs/hotpotqa/$MODEL_SUBDIR"

# For regular experiments, use the below
# INDIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/experiment_outputs/$MODEL_SUBDIR"
# OUT_DIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/evaluation_outputs/$MODEL_SUBDIR"

mkdir -p logs "$OUT_DIR"

EXPERIMENT_DIR="$INDIR"

# Use the current generator model as the same-answer judge model by default.
export SAME_ANSWER_MODEL_NAME="$MODEL_SUBDIR"

# for hotpot, use below
# hotpot_search hotpot_replace_one hotpot_replace_all

# for rerank only needs: hotpot_rerank_lambda0.1

# for regular: replace_all replace_one search

for base in hotpot_rerank_lambda1.0; do
  in_file="$EXPERIMENT_DIR/${base}.json"
  out_file="$OUT_DIR/${base}_eval.json"
  if [[ -f "$in_file" ]]; then
    python -u evaluation.py "$in_file" "$out_file" --cache-dir "$CACHE_DIR"
  else
    echo "Skipping missing experiment file: $in_file"
  fi
done
