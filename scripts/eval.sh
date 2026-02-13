#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=logs/evaluation_%A.out
#SBATCH --error=logs/evaluation_%A.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -C "vram40|vram48&sm_70|sm_75|sm_80|sm_86|sm_89|sm_90"
#SBATCH --cpus-per-task=2

module load conda/latest
conda activate ragenv

EXPERIMENT_NAME=replace_all_full_dataset

OUTPUT_FILE_PATH="experiment_outputs/$EXPERIMENT_NAME.json"
RESULTS_FILE_PATH="evaluation_outputs/$EXPERIMENT_NAME.json"

python -u evaluation.py $OUTPUT_FILE_PATH $RESULTS_FILE_PATH
