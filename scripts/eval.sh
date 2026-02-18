#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=logs/evaluation_%A.out
#SBATCH --error=logs/evaluation_%A.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -C "vram40|vram48|vram80"
#SBATCH --cpus-per-task=2

module load conda/latest
conda activate ragenv

module load cuda/12.6

nvidia-smi

python -u evaluation.py experiment_outputs/local_search.json evaluation_outputs/local_search_eval.json
