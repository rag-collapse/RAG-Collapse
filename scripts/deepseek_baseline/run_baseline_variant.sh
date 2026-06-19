#!/bin/bash
# One BASELINE variant of the graphite (umass-entity) RAG-collapse loop for
# DeepSeek-R1-Distill-Qwen-7B, as a CPU client talking to two SHARED vLLM servers
# (DeepSeek answer + Qwen2.5-7B doc). Reproduces the canonical baseline config that
# produced the other models' local_<variant>.json files. Launched by launch_deepseek_baseline.sh.
#
#SBATCH -J ds-baseline-cli
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=48g
#SBATCH -t 47:00:00
#SBATCH -o logs/ds_baseline_%A.out
#SBATCH -e logs/ds_baseline_%A.err
#SBATCH --mail-type=END,FAIL
#
# 47h (< the servers' 48h) so the shared servers always outlive the client.
# Required env (set by the launcher): VLLM_API_BASE, DOC_VLLM_API_BASE, VARIANT.
# VARIANT is the OUTPUT label: replace_all | replace_one | search.
set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR" || exit 1; fi

module load conda/latest
conda activate ragenv

SCR=/scratch4/workspace/oyilmazel_umass_edu-rag_collapse
export HF_HOME="${HF_HOME:-$SCR/hf_cache}"
export HF_HUB_CACHE="$HF_HOME"
export CUDA_VISIBLE_DEVICES=""        # server-mode client; search embeddings (MiniLM) run on CPU
mkdir -p logs

: "${VLLM_API_BASE:?set VLLM_API_BASE (shared DeepSeek answer server)}"
: "${DOC_VLLM_API_BASE:?set DOC_VLLM_API_BASE (shared Qwen2.5-7B doc server)}"

MODEL="${MODEL:-deepseek-r1-distill-qwen-7b}"      # == answer server --served-model-name
DOC_MODEL="${DOC_MODEL:-qwen2.5-7b-doc}"           # == doc server --served-model-name (baseline name)
DATASET="${DATASET:-datasets/umass_data.entity.chatgpt.400.jsonl}"   # graphite 400-question set
MAX_Q="${MAX_Q:-400}"
NUM_RUNS="${NUM_RUNS:-10}"                          # baseline default
CHARS_PER_DOC="${CHARS_PER_DOC:-400}"              # baseline default
# DeepSeek-R1 is a reasoning model: it needs answer-token headroom for its <think> trace
# (the non-reasoning baselines used 512). Doc length pinned to 512 to match the other models.
MAX_TOKENS="${MAX_TOKENS:-4096}"
DOC_MAX_TOKENS="${DOC_MAX_TOKENS:-512}"
VARIANT="${VARIANT:-search}"
# Store into the consolidated all_experiments tree, matching the other baselines' layout:
#   <ALL_EXP_BASE>/graphite/baseline/<variant>/experiment_outputs/<org>/<model>/local_<variant>.json
ALL_EXP_BASE="${ALL_EXP_BASE:-/work/pi_dagarwal_umass_edu/project_4/file_storage/all_experiments}"
MODEL_SUBDIR="${MODEL_SUBDIR:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"

# map the output label -> pipeline-variant + variant-specific flags (from baseline metadata)
case "$VARIANT" in
  replace_all)  PV="hybrid";       EXTRA="--num-synth-docs 10 --num-db-docs 0" ;;
  replace_one)  PV="replace_one";  EXTRA="" ;;
  search)       PV="search";       EXTRA="--search-top-k 10 --search-chunk-size 500 --search-chunk-overlap 50" ;;
  *) echo "unknown VARIANT '$VARIANT' (expected replace_all|replace_one|search)"; exit 1 ;;
esac
OUTDIR="$ALL_EXP_BASE/graphite/baseline/$VARIANT/experiment_outputs/$MODEL_SUBDIR"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/local_$VARIANT.json"

echo "=== DeepSeek baseline: VARIANT=$VARIANT (pipeline-variant=$PV) -> $OUT ==="
python -u pipeline.py \
  --model-mode server --vllm-api-base "$VLLM_API_BASE" --model-name "$MODEL" \
  --doc-model-mode server --doc-vllm-api-base "$DOC_VLLM_API_BASE" --doc-model-name "$DOC_MODEL" \
  --dataset-path "$DATASET" --num-runs "$NUM_RUNS" --chars-per-doc "$CHARS_PER_DOC" \
  --max-questions "$MAX_Q" --max-tokens "$MAX_TOKENS" --doc-max-tokens "$DOC_MAX_TOKENS" \
  --pipeline-variant "$PV" $EXTRA \
  --output-path "$OUT"

echo "=== Done VARIANT=$VARIANT -> $OUT ==="
