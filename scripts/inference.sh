#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=logs/inference_%A.out
#SBATCH --error=logs/inference_%A.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -C "vram40|vram48&sm_70|sm_75|sm_80|sm_86|sm_89|sm_90"
#SBATCH --cpus-per-task=2

module load conda/latest
module load cuda/12.6

nvidia-smi

# conda environment name
conda activate ragenv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u llm_service/inference_example.py

# you can observe status of jobs with squeue --me
