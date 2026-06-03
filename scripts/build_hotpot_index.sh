#!/bin/bash
#SBATCH --job-name=index
#SBATCH --output=logs/index_%A.out
#SBATCH --time=48:00:00
#SBATCH --partition=gpu,superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40|vram48|vram80
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gpu013,gpu016,
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=pi_hzamani_umass_edu

module load conda/latest
module load cuda/12.6

conda activate rag

python build_hotpotqa_index.py
