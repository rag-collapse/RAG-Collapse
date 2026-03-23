#!/bin/bash
# --- SLURM (tuned for 14B model on 2 GPUs) ---
#SBATCH --job-name=pipeline
#SBATCH --output=logs/pipeline_%A.out
#SBATCH --error=logs/pipeline_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40|vram48|vram80
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL

# Run from submit dir so pipeline.py and paths resolve
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
fi

# --- Conda ---
module load conda/latest
conda activate ragenv

# module load cuda/12.6

# nvidia-smi

# Ensure a valid cache dir for vLLM/HF (avoids FileNotFoundError in weight_utils.get_lock)
CACHE_DIR="/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache/"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"

if [[ -z "$VLLM_API_BASE" ]]; then
  echo "ERROR: VLLM_API_BASE is not set. Start the vLLM server first, then:"
  echo "  export VLLM_API_BASE=\"http://<fqdn>:5150/v1\""
  echo "  sbatch --export=ALL scripts/pipeline.sh"
  exit 1
fi
echo "Using VLLM_API_BASE=$VLLM_API_BASE"

# --- Config ---
DATASET="datasets/umass_data.entity.chatgpt.400.jsonl"
MODEL="Qwen/Qwen2.5-14B-Instruct"
OUTDIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/experiment_outputs/$MODEL"

COMMON="--dataset-path $DATASET --chars-per-doc 400 --num-runs 10"
EXTRA="--max-questions 400"

mkdir -p logs "$OUTDIR"

# --- Server mode: HTTP client to a running vLLM server (recommended) ---
# Start vLLM server first, then export VLLM_API_BASE and sbatch --export=ALL
# run_server() {
#   python -u pipeline.py --model-mode server --vllm-api-base "$VLLM_API_BASE" --model-name "$MODEL" $COMMON $EXTRA "$@"
# }

# To use a separate (smaller) model for document generation, start a second
# vLLM server on a different port and export DOC_VLLM_API_BASE, then replace
# the run_server definition above with this one:
run_server() {
  python -u pipeline.py --model-mode server --vllm-api-base "$VLLM_API_BASE" \
    --doc-model-mode server --doc-vllm-api-base "$DOC_VLLM_API_BASE" \
    --model-name "$MODEL" $COMMON $EXTRA "$@"
}

# --- Smoke test (1 question, 2 rounds) ---
# Uncomment the block below to quickly verify the server connection and pipeline logic.
# Results go to experiment_outputs/smoke_test/ so they won't overwrite production outputs.
# Once confirmed working, comment this block out and uncomment the production runs below.
#
# mkdir -p experiment_outputs/smoke_test
# run_server --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 \
#   --num-iterations 2 --max-questions 1 \
#   --output-path experiment_outputs/smoke_test/replace_all.json
#
# run_server --pipeline-variant replace_one --num-iterations 2 --max-questions 1 \
#   --output-path experiment_outputs/smoke_test/replace_one.json
#
# run_server --pipeline-variant search --num-iterations 2 --max-questions 1 \
#   --output-path experiment_outputs/smoke_test/search.json
#
# run_server --pipeline-variant agentic_rag --num-iterations 2 --max-questions 1 \
#   --output-path experiment_outputs/smoke_test/agentic_rag.json

# --- Production runs ---
# I would suggest running each pipeline variant separately to ensure clear logs, and if one fails it won't compromise the others. You can comment/uncomment the blocks below as needed.
# Replace All, Replace One, Search
#run_server --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 --num-iterations 10 --output-path "$OUTDIR/local_replace_all.json"

#run_server --pipeline-variant replace_one --num-iterations 20 --output-path "$OUTDIR/local_replace_one.json"

#run_server --pipeline-variant search --num-iterations 30 --output-path "$OUTDIR/local_search.json"

#run_server --pipeline-variant agentic_rag --num-iterations 30 --output-path "$OUTDIR/local_agentic_rag.json"


# --- Local mode: in-process vLLM (needs GPU allocation in this job) ---
# Requires --gres=gpu:2 and -C "vram40|vram48" in SLURM headers above.
# run_local() {
#   python -u pipeline.py --model-mode local --model-name "$MODEL" $COMMON $EXTRA --tensor-parallel-size 2 "$@"
# }
# run_local --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 --num-iterations 10 --output-path "$OUTDIR/local_replace_all.json"
# run_local --pipeline-variant replace_one --num-iterations 20 --output-path "$OUTDIR/local_replace_one.json"
# run_local --pipeline-variant search --num-iterations 30 --output-path "$OUTDIR/local_search.json"


# --- API mode (uncomment and set API_KEY; comment out server block above) ---
# MODEL_API="openai/gpt4o"
# run_api() { python -u pipeline.py --model-mode api --model-name "$MODEL_API" $COMMON $EXTRA "$@"; }
# run_api --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 --output-path "$OUTDIR/api_hybrid_replace_all.json"
# run_api --pipeline-variant hybrid --num-synth-docs 1 --num-db-docs 3 --output-path "$OUTDIR/api_hybrid_replace_one.json"
# run_api --pipeline-variant search --output-path "$OUTDIR/api_search.json"

# --- API agentic_rag smoke test (1 question, 2 iterations) ---
# Requires API_KEY set in .env and MODEL_API pointing to an OpenAI-compatible model.
# MODEL_API="openai/gpt4o"
# run_api() { python -u pipeline.py --model-mode api --model-name "openai/gpt-4o" $COMMON "$@"; }
# mkdir -p experiment_outputs/smoke_test
# run_api --pipeline-variant agentic_rag --num-iterations 2 --max-questions 1 \
#   --output-path experiment_outputs/smoke_test/api_agentic_rag.json

# --- API agentic_rag production run ---
# run_api --pipeline-variant agentic_rag --num-iterations 30 --max-questions 400 \
#   --output-path "$OUTDIR/api_agentic_rag.json"
