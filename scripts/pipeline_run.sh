#!/bin/bash
set -e

module load conda/latest
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ragenv

python -m pip install -r requirements.txt

# -------------------------
# API mode (CLI args)
# -------------------------
export API_KEY= # set this to your OpenAI API key
 
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
