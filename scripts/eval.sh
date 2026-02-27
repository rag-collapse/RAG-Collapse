#!/bin/bash
# --- SLURM ---
#SBATCH --job-name=evaluation
#SBATCH --output=logs/evaluation_%A.out
#SBATCH --error=logs/evaluation_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -C "vram40|vram48|vram80"
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL

# --- Conda ---
module load conda/latest
conda activate ragenv

module load cuda/12.6

nvidia-smi

# --- Config (edit as needed) ---
INPUT_SUBDIR="${INPUT_SUBDIR:-qwen-14b}"
INDIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/experiment_outputs/$INPUT_SUBDIR"
OUTDIR="evaluation_outputs/$INPUT_SUBDIR"
MODEL="Qwen/Qwen2.5-7B-Instruct"

mkdir -p logs "$OUTDIR"

run_eval() {
  python -u evaluation.py "$@"
}

# --- Evaluate each variant ---
run_eval "$INDIR/local_replace_all.json" "$OUTDIR/local_replace_all_eval.json"
run_eval "$INDIR/local_replace_one.json" "$OUTDIR/local_replace_one_eval.json"
run_eval "$INDIR/local_search.json" "$OUTDIR/local_search_eval.json"
