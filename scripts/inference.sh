#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=logs/inference_%A.out
#SBATCH --error=logs/inference_%A.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

module load conda/latest
conda activate ragenv

# Requires a running vLLM server. Start it first:
#   sbatch scripts/start_llm_server.sh
# Then get the URL from the log and export it before submitting this job:
#   export VLLM_API_BASE="http://<fqdn>:5150/v1"
#   sbatch --export=ALL scripts/inference.sh
if [[ -z "$VLLM_API_BASE" ]]; then
  echo "ERROR: VLLM_API_BASE is not set. Start the vLLM server first and export the URL."
  exit 1
fi

echo "Using vLLM server at: $VLLM_API_BASE"

python -u llm_service/inference_example.py

# you can observe status of jobs with squeue --me
