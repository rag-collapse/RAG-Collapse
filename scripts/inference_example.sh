#!/bin/bash
#SBATCH --job-name=inference_example
#SBATCH --output=logs/inference_example_%A_%a.out
#SBATCH --error=logs/inference_example_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -C "vram40|vram48&sm_70|sm_75|sm_80|sm_86|sm_89|sm_90"
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=oyilmazel@umass.edu

module load conda/latest
module load cuda/12.6

nvidia-smi

# You will change this to your own conda environment name
conda activate ragenv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u llm_service/inference_example.py
