#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=logs/pipeline_run_%A.out
#SBATCH --error=logs/pipeline_run_%A.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -C "vram40|vram48&sm_70|sm_75|sm_80|sm_86|sm_89|sm_90"
#SBATCH --cpus-per-task=2

set -e

module load conda/latest
conda activate ragenv

python -m pip install -r requirements.txt

# -------------------------
# API mode (CLI args)
# -------------------------
export API_KEY= # set this to LiteLLM API key
 
python -m pipeline.run_pipeline \
  --model-mode api \
  --model-name openai/gpt4o \
  --dataset-path datasets/umass_data.entity.chatgpt.50.jsonl \
  --output-path experiments_output.json \
  --max-questions 1 \
  --num-iterations 5 \
  --num-runs 10 \
  --chars-per-doc 400

# -------------------------
# Local mode (uncomment to use)
# -------------------------
# python -m pipeline.run_pipeline \
#   --model-mode local \
#   --model-name Qwen/Qwen2.5-1.5B-Instruct \
#   --dataset-path datasets/umass_data.entity.chatgpt.50.jsonl \
#   --output-path example_experiments_output.json \
#   --max-questions 1 \
#   --num-iterations 5 \
#   --num-runs 10 \
#   --chars-per-doc 400
