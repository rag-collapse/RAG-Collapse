#!/bin/bash
# --- SLURM ---
#SBATCH --job-name=evaluation
#SBATCH --output=logs/evaluation_%A.out
#SBATCH --error=logs/evaluation_%A.err
#SBATCH --time=8:00:00
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
INDIR="experiment_outputs"
OUTDIR="evaluation_outputs"
MODEL="Qwen/Qwen2.5-14B-Instruct"

mkdir -p logs "$OUTDIR"

run_eval() {
  python -u evaluation.py "$@"
}

# --- Evaluate each variant ---
run_eval "$INDIR/${MODEL}_local_replace_all.json" "$OUTDIR/${MODEL}_local_replace_all_eval.json"
run_eval "$INDIR/${MODEL}_local_replace_one.json" "$OUTDIR/${MODEL}_local_replace_one_eval.json"
run_eval "$INDIR/${MODEL}_local_search_test.json" "$OUTDIR/${MODEL}_local_search_test_eval.json"
