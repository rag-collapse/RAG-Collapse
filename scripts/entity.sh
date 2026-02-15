#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=logs/pipeline_%A.out
#SBATCH --error=logs/pipeline_%A.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -C "vram40|vram48&sm_70|sm_75|sm_80|sm_86|sm_89|sm_90"
#SBATCH --cpus-per-task=2

module load conda/latest
conda activate ragenv

# API mode(uncomment to use)

# python -u entity_extraction.py \
#   --model-mode api \
#   --model-name gpt-4o \
#   --experiment-files experiment_outputs/local_search.json \
#                      experiment_outputs/local_replace_one.json \
#                      experiment_outputs/local_replace_all.json \
#   --output-dir entity_extraction_output


# Local mode (used by default)

python -u entity_extraction.py \
  --model-mode local \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --experiment-files experiment_outputs/local_search.json \
                     experiment_outputs/local_replace_one.json \
                     experiment_outputs/local_replace_all.json \
  --output-dir entity_extraction_output
