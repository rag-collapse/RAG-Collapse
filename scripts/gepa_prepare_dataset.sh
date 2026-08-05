#!/bin/bash
# --- SLURM ---
#SBATCH --job-name=gepa_prepare_dataset
#SBATCH --output=logs/gepa_prepare_dataset_%A.out
#SBATCH --error=logs/gepa_prepare_dataset_%A.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -C "vram48|vram80"
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rsenapati@umass.edu

# Builds gepa_optimization/data/{train,val,test}.jsonl by:
#   1. Loading 50 questions from datasets/umass_data.entity.chatgpt.400.jsonl
#   2. Encoding the mteb/hotpotqa corpus (~5M passages) with intfloat/e5-small-v2
#   3. Building a FAISS IndexIVFFlat index (saved for reuse)
#   4. Retrieving top-10 docs per question, splitting 60/20/20
#
# The FAISS index is saved to INDEX_DIR and reused on subsequent runs.
# Run this once before submitting gepa_optimization.sh.

set -eo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
fi

# --- Conda ---
module load conda/latest
conda activate ragenv

module load cuda/12.6

nvidia-smi

# --- HF cache ---
CACHE_DIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hf_cache/"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"

# FAISS index is large — store on scratch to avoid re-encoding the full corpus
INDEX_DIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hotpotqa_index"
mkdir -p "$INDEX_DIR" logs

echo "========================================"
echo "GEPA Dataset Preparation"
echo "========================================"
echo "HF cache  : $CACHE_DIR"
echo "FAISS index dir: $INDEX_DIR"
echo "========================================"

python -u gepa_optimization/prepare_dataset.py \
  --cache-dir "$CACHE_DIR" \
  --index-dir "$INDEX_DIR"

echo ""
echo "Dataset written to gepa_optimization/data/"
echo "Next: sbatch scripts/gepa_optimization.sh"
