#!/bin/bash
# --- SLURM (tuned for 14B model on 2 GPUs) ---
#SBATCH --job-name=pipeline
#SBATCH --output=logs/pipeline_%A.out
#SBATCH --error=logs/pipeline_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --mem=8g
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL

# Run from submit dir so pipeline.py and paths resolve
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
fi

# --- Conda ---
module load conda/latest
conda activate ragenv

module load cuda/12.6

nvidia-smi

# Ensure a valid cache dir for vLLM/HF (avoids FileNotFoundError in weight_utils.get_lock)
CACHE_DIR="/work/pi_dagarwal_umass_edu/hf_cache/"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"
export VLLM_API_BASE="http://gypsum-gpu188.unity.rc.umass.edu:5150/v1"

# --- Config (14B model, 2 GPUs: tensor-parallel-size 2) ---
DATASET="datasets/umass_data.entity.chatgpt.400.jsonl"
#OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-qwen-14B}"
OUTDIR="experiment_outputs"
MODEL="Qwen/Qwen2.5-14B-Instruct"
#TPARALLEL=2
COMMON="--dataset-path $DATASET --num-runs 10 --chars-per-doc 400 --tensor-parallel-size $TPARALLEL"
#EXTRA="--max-questions 50"

mkdir -p logs "$OUTDIR"

# Ensure CUDA_VISIBLE_DEVICES is set (SLURM sets it automatically with --gres, but verify for multi-GPU)
if [ "$TPARALLEL" -gt 1 ] && [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  echo "WARNING: TPARALLEL=$TPARALLEL but CUDA_VISIBLE_DEVICES not set. SLURM should set this with --gres=gpu:$TPARALLEL"
fi

# vLLM with tensor_parallel_size > 1 spawns its own processes; using torch.distributed.run causes conflicts
run_local() {
  python -u pipeline.py --model-mode local --model-name "$MODEL" $COMMON $EXTRA "$@"
}

# I would suggest running each pipeline variant separately to ensure clear logs, and if one fails it won't compromise the others. You can comment/uncomment the blocks below as needed.
# Replace All, Replace One, Search
#run_local --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0  --num-iterations 10 --output-path "$OUTDIR/local_replace_all.json"

run_local --pipeline-variant replace_one  --num-runs 20 --output-path "$OUTDIR/${MODEL}_local_replace_one.json"

#run_local --pipeline-variant search  --num-runs 30 --output-path "$OUTDIR/${MODEL}_local_search_test.json"

# --- API mode (uncomment and set API_KEY; comment out Local block above) ---
# MODEL_API="openai/gpt4o"
# run_api() { python -u pipeline.py --model-mode api --model-name "$MODEL_API" $COMMON $EXTRA "$@"; }
# run_api --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 --output-path "$OUTDIR/api_hybrid_replace_all.json"
# run_api --pipeline-variant hybrid --num-synth-docs 1 --num-db-docs 3 --output-path "$OUTDIR/api_hybrid_replace_one.json"
# run_api --pipeline-variant search --output-path "$OUTDIR/api_search.json"
