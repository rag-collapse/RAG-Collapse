#!/bin/bash
# One variant of the CROSS-MODEL baseline on the graphite (umass-entity) RAG-collapse loop.
# CPU client talking to THREE shared vLLM servers (one per role): a MAIN answer model, a SIDE
# answer model, and a doc-gen model. Both models answer from the same context each round, but the
# next round's documents are synthesized from the SIDE model's answers (--docs-from-side); the MAIN
# model reads that stream and is what we measure. Launched by launch_cross_model_baseline.sh.
#
#SBATCH -J xmodel-cli
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=24g
#SBATCH -t 47:00:00
#SBATCH -o logs/xmodel_%A.out
#SBATCH -e logs/xmodel_%A.err
#SBATCH --mail-type=END,FAIL
#
# Required env (set by the launcher): VLLM_API_BASE (main), SIDE_VLLM_API_BASE (side),
# DOC_VLLM_API_BASE (doc-gen), VARIANT (output label: replace_all|replace_one|search).
set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR" || exit 1; fi

module load conda/latest
conda activate ragenv

SCR=/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu
export HF_HOME="${HF_HOME:-$SCR/hf_cache}"
export HF_HUB_CACHE="$HF_HOME"
export CUDA_VISIBLE_DEVICES=""        # server-mode client; search embeddings (MiniLM) run on CPU
mkdir -p logs

: "${VLLM_API_BASE:?set VLLM_API_BASE (shared MAIN answer server)}"
: "${SIDE_VLLM_API_BASE:?set SIDE_VLLM_API_BASE (shared SIDE answer server)}"
: "${DOC_VLLM_API_BASE:?set DOC_VLLM_API_BASE (shared doc-gen server)}"

MODEL="${MODEL:-qwen2.5-14b}"                       # == main answer server --served-model-name (measured)
SIDE_MODEL="${SIDE_MODEL:-deepseek-r1-distill-qwen-7b}"  # == side answer server --served-model-name (writes docs)
DOC_MODEL="${DOC_MODEL:-qwen2.5-7b-doc}"            # == doc-gen server --served-model-name
DATASET="${DATASET:-datasets/umass_data.entity.chatgpt.400.jsonl}"   # graphite 400-question set
MAX_Q="${MAX_Q:-400}"
NUM_RUNS="${NUM_RUNS:-10}"
CHARS_PER_DOC="${CHARS_PER_DOC:-400}"
MAX_TOKENS="${MAX_TOKENS:-512}"                     # main = Qwen2.5-14B (non-reasoning): 512 is plenty
SIDE_MAX_TOKENS="${SIDE_MAX_TOKENS:-4096}"          # side = DeepSeek-R1 (reasoning): needs <think> headroom
DOC_MAX_TOKENS="${DOC_MAX_TOKENS:-512}"
MAX_ITERS="${MAX_ITERS:-2}"                         # rounds per variant (this run: 2)
VARIANT="${VARIANT:-search}"
# Store into the all_experiments tree keyed by MAIN model (top) then SIDE model then variant:
#   <ALL_EXP_BASE>/graphite/cross-model-baseline/<main-org>/<main-model>/<side>/<variant>/experiment_outputs/local_<variant>.json
ALL_EXP_BASE="${ALL_EXP_BASE:-/work/pi_dagarwal_umass_edu/project_4/file_storage/all_experiments}"
MODEL_SUBDIR="${MODEL_SUBDIR:-Qwen/Qwen2.5-14B-Instruct}"   # the MAIN (measured) model

# map the output label -> pipeline-variant + variant-specific flags (same as the baseline)
case "$VARIANT" in
  replace_all)  PV="hybrid";       EXTRA="--num-synth-docs 10 --num-db-docs 0" ;;
  replace_one)  PV="replace_one";  EXTRA="" ;;
  search)       PV="search";       EXTRA="--search-top-k 10 --search-chunk-size 500 --search-chunk-overlap 50" ;;
  *) echo "unknown VARIANT '$VARIANT' (expected replace_all|replace_one|search)"; exit 1 ;;
esac
OUTDIR="$ALL_EXP_BASE/graphite/cross-model-baseline/$MODEL_SUBDIR/$SIDE_MODEL/$VARIANT/experiment_outputs"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/local_$VARIANT.json"

echo "=== cross-model: VARIANT=$VARIANT (pv=$PV) main=$MODEL side=$SIDE_MODEL doc=$DOC_MODEL rounds=$MAX_ITERS -> $OUT ==="
python -u pipeline.py \
  --model-mode server --vllm-api-base "$VLLM_API_BASE" --model-name "$MODEL" \
  --side-model-mode server --side-vllm-api-base "$SIDE_VLLM_API_BASE" --side-model-name "$SIDE_MODEL" \
  --side-max-tokens "$SIDE_MAX_TOKENS" --docs-from-side \
  --doc-model-mode server --doc-vllm-api-base "$DOC_VLLM_API_BASE" --doc-model-name "$DOC_MODEL" \
  --dataset-path "$DATASET" --num-runs "$NUM_RUNS" --chars-per-doc "$CHARS_PER_DOC" \
  --max-questions "$MAX_Q" --max-tokens "$MAX_TOKENS" --doc-max-tokens "$DOC_MAX_TOKENS" \
  --max-iterations "$MAX_ITERS" \
  --pipeline-variant "$PV" $EXTRA \
  --output-path "$OUT"

echo "=== Done VARIANT=$VARIANT -> $OUT ==="
