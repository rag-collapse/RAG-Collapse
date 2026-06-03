#!/bin/bash
#SBATCH --job-name=gepa-llama3.1-8b
#SBATCH --output=logs/gepa_llama3.1_8b_%A.out
#SBATCH --error=logs/gepa_llama3.1_8b_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0          # CPU-only job: connects to a running vLLM server
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL

# ── Fill in these two URLs before submitting ──────────────────────────────────
VLLM_API_BASE="http://<llama-fqdn>:5155/v1"         # server_llama3.1_8b.sh
DOC_VLLM_API_BASE="http://<docgen-fqdn>:5153/v1"    # server_qwen7b_docgen.sh
# ─────────────────────────────────────────────────────────────────────────────

set -eo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
fi

module load conda/latest
conda activate ragenv

MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUTDIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/experiment_outputs/gepa/${MODEL}"
DATASET="datasets/umass_data.entity.chatgpt.400.jsonl"

COMMON="--model-mode server --vllm-api-base $VLLM_API_BASE
        --doc-model-mode server --doc-vllm-api-base $DOC_VLLM_API_BASE
        --model-name $MODEL
        --dataset-path $DATASET
        --chars-per-doc 400 --num-runs 10 --max-questions 400
        --use-gepa-prompt"

mkdir -p logs "$OUTDIR"

echo "Task server : $VLLM_API_BASE"
echo "Doc server  : $DOC_VLLM_API_BASE"

echo "=== replace_all ===" && \
python -u pipeline.py $COMMON \
  --pipeline-variant hybrid \
  --num-synth-docs 10 --num-db-docs 0 \
  --num-iterations 10 \
  --output-path "$OUTDIR/local_replace_all.json"

echo "=== replace_one ===" && \
python -u pipeline.py $COMMON \
  --pipeline-variant replace_one \
  --num-iterations 20 \
  --output-path "$OUTDIR/local_replace_one.json"

echo "=== search ===" && \
python -u pipeline.py $COMMON \
  --pipeline-variant search \
  --num-iterations 30 \
  --output-path "$OUTDIR/local_search.json"
