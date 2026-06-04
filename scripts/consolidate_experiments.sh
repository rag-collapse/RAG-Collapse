#!/bin/bash
#SBATCH -J consolidate-data
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=8g
#SBATCH --nodes=1
#SBATCH -t 02:00:00
#SBATCH -o logs/consolidate-%j.out
#SBATCH -e logs/consolidate-%j.err
#SBATCH --mail-type=END,FAIL

# Consolidates every RAG-collapse experiment JSON under
# /work/.../file_storage/all_experiments/<dataset>/<variant>/<output_tree>/<owner>/...
# Pure filesystem copy (stdlib only) — no GPU needed.

set -euo pipefail
mkdir -p logs

module load conda/latest
conda activate ragenv

# SLURM copies the batch script to a spool dir, so resolve relative to the
# submit directory instead of ${BASH_SOURCE[0]}.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "Host: $(hostname)   Start: $(date)"
python3 "${SCRIPT_DIR}/consolidate_experiments.py"
echo "Done: $(date)"
