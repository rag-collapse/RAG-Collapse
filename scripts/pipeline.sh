#!/bin/bash
#SBATCH --job-name=pipeline
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

EXPERIMENT_NAME=replace_all_full_dataset

MODEL_MODE=local # or api
MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
DATASET_PATH=datasets/umass_data.entity.chatgpt.50.jsonl
OUTPUT_PATH=experiment_outputs/$EXPERIMENT_NAME.json
MAX_QUESTIONS=50
NUM_ITERATIONS=4
NUM_RUNS=5
CHARS_PER_DOC=800

python -u pipeline.py \
  --model-mode $MODEL_MODE \
  --model-name $MODEL_NAME \
  --dataset-path $DATASET_PATH \
  --output-path $OUTPUT_PATH \
  --max-questions $MAX_QUESTIONS \
  --num-iterations $NUM_ITERATIONS \
  --num-runs $NUM_RUNS \
  --chars-per-doc $CHARS_PER_DOC
