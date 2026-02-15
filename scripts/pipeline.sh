#!/bin/bash
# --- SLURM ---
#SBATCH --job-name=pipeline
#SBATCH --output=logs/pipeline_%A.out
#SBATCH --error=logs/pipeline_%A.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -C "vram40|vram48&sm_70|sm_75|sm_80|sm_86|sm_89|sm_90"
#SBATCH --cpus-per-task=2

# --- Conda ---
module load conda/latest
conda activate ragenv

# Ensure a valid cache dir for vLLM/HF (avoids FileNotFoundError in weight_utils.get_lock)
CACHE_DIR="$(pwd)/model_cache"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"

# --- Config (edit as needed) ---
DATASET="datasets/umass_data.entity.chatgpt.50.jsonl"
OUTDIR="experiment_outputs"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
COMMON="--dataset-path $DATASET --num-runs 10 --chars-per-doc 400"
EXTRA="--max-questions 8"

mkdir -p logs "$OUTDIR"

# --- Local mode (GPU): paper variants only ---
run_local() {
  python -u pipeline.py --model-mode local --model-name "$MODEL" $COMMON $EXTRA "$@"
}

# Replace All, Replace One, Search
run_local --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 \
  --output-path "$OUTDIR/local_replace_all.json"

run_local --pipeline-variant replace_one --output-path "$OUTDIR/local_replace_one.json"

run_local --pipeline-variant search --output-path "$OUTDIR/local_search.json"

# --- API mode (uncomment and set API_KEY; comment out Local block above) ---
# MODEL_API="openai/gpt4o"
# run_api() { python -u pipeline.py --model-mode api --model-name "$MODEL_API" $COMMON $EXTRA "$@"; }
# run_api --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 --output-path "$OUTDIR/api_hybrid_replace_all.json"
# run_api --pipeline-variant hybrid --num-synth-docs 1 --num-db-docs 3 --output-path "$OUTDIR/api_hybrid_replace_one.json"
# run_api --pipeline-variant search --output-path "$OUTDIR/api_search.json"
