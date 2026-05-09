#!/bin/bash
# --- SLURM ---
#SBATCH --job-name=gepa_optimization
#SBATCH --output=logs/gepa_optimization_%A.out
#SBATCH --error=logs/gepa_optimization_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rsenapati@umass.edu

# GEPA prompt optimization — no GPU required (all LLM calls go to keymaker API).
# The search simulation variant uses all-MiniLM-L6-v2 locally on CPU (< 100 MB).
#
# Prerequisites:
#   1. Generate the dataset first if data/train.jsonl does not exist:
#        sbatch scripts/gepa_prepare_dataset.sh
#      OR uncomment the "Data prep" block below to run it inline.
#   2. Export your keymaker API key:
#        export API_KEY="your-key-here"
#        sbatch --export=ALL scripts/gepa_optimization.sh

set -eo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
fi

# --- Conda ---
module load conda/latest
conda activate ragenv

# --- HF cache (avoids re-downloading models on every run) ---
CACHE_DIR="/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache/"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"
export API_KEY="REDACTED"

# --- API key check ---
if [[ -z "${API_KEY:-}" ]]; then
  echo "ERROR: API_KEY is not set."
  echo "  export API_KEY='your-keymaker-key'"
  echo "  sbatch --export=ALL scripts/gepa_optimization.sh"
  exit 1
fi

# --- Config (override via environment before submitting) ---
LITELLM_API_BASE="${LITELLM_API_BASE:-https://thekeymaker.umass.edu/}"
TASK_MODEL="${TASK_MODEL:-openai/claude-haiku-4-5}"
DOC_GEN_MODEL="${DOC_GEN_MODEL:-openai/gemma-3-12b-it}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt4o}"
REFLECTION_MODEL="${REFLECTION_MODEL:-openai/claude-opus-4-1}"
EMBED_MODEL="${EMBED_MODEL:-all-MiniLM-L6-v2}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-300}"
GEPA_RUN_DIR="${GEPA_RUN_DIR:-gepa_runs/rag_system_prompt_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"

export LITELLM_API_BASE TASK_MODEL DOC_GEN_MODEL JUDGE_MODEL REFLECTION_MODEL EMBED_MODEL MAX_METRIC_CALLS GEPA_RUN_DIR

echo "========================================"
echo "GEPA Optimization"
echo "========================================"
echo "API base        : $LITELLM_API_BASE"
echo "Task model      : $TASK_MODEL"
echo "Doc gen model   : $DOC_GEN_MODEL"
echo "Judge model     : $JUDGE_MODEL"
echo "Reflection model: $REFLECTION_MODEL"
echo "Embed model     : $EMBED_MODEL"
echo "Max metric calls: $MAX_METRIC_CALLS"
echo "Run dir         : $GEPA_RUN_DIR"
echo "========================================"

mkdir -p logs gepa_runs

# --- Optional: data prep inline (uncomment if dataset not yet generated) ---
# Requires GPU partition for fast E5 corpus encoding; see gepa_prepare_dataset.sh
# for a dedicated GPU data-prep job.
#
# DATA_DIR="gepa_optimization/data"
# if [[ ! -f "$DATA_DIR/train.jsonl" ]]; then
#   echo "Dataset not found — running prepare_dataset.py ..."
#   INDEX_DIR="/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpotqa_index"
#   mkdir -p "$INDEX_DIR"
#   python -u gepa_optimization/prepare_dataset.py \
#     --cache-dir "$CACHE_DIR" \
#     --index-dir "$INDEX_DIR"
# else
#   echo "Dataset found at $DATA_DIR — skipping prepare_dataset.py"
# fi

# --- Dataset check ---
DATA_DIR="gepa_optimization/data"
for split in train val test; do
  if [[ ! -f "$DATA_DIR/${split}.jsonl" ]]; then
    echo "ERROR: Missing $DATA_DIR/${split}.jsonl"
    echo "  Run: sbatch scripts/gepa_prepare_dataset.sh"
    exit 1
  fi
done
echo "Dataset: $(wc -l < "$DATA_DIR/train.jsonl") train | $(wc -l < "$DATA_DIR/val.jsonl") val | $(wc -l < "$DATA_DIR/test.jsonl") test"

# --- Run GEPA optimization ---
python -u gepa_optimization/run_optimization.py

echo ""
echo "Optimization complete. Checkpoints saved to gepa_runs/rag_system_prompt/"
echo "To apply the best prompt:"
echo "  python gepa_optimization/apply_best_prompt.py --prompt '<prompt text>'"
