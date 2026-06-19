#!/bin/bash
# CPU client: base HotpotQA recursive-RAG loop (faithful synthesis only) over a sweep of
# round-0 distractor fractions, INCLUDING fraction 0 = the matched no-distractor baseline.
# Talks to two SHARED vLLM servers (answer + Qwen2.5-7B doc). Launched by launch.sh.
#
#SBATCH -J base-distractor-sweep
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=96g
#SBATCH -t 47:00:00
#SBATCH -o logs/base_distractor_%A.out
#SBATCH -e logs/base_distractor_%A.err
#SBATCH --mail-type=END,FAIL
set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR" || exit 1; fi

module load conda/latest
conda activate ragenv

SCR=/scratch4/workspace/oyilmazel_umass_edu-rag_collapse
export HF_HOME="${HF_HOME:-$SCR/hf_cache}"
export HF_HUB_CACHE="$HF_HOME"
export CUDA_VISIBLE_DEVICES=""          # server-mode client; E5/MiniLM embeddings run on CPU
mkdir -p logs

: "${VLLM_API_BASE:?set VLLM_API_BASE (shared answer server)}"
: "${DOC_VLLM_API_BASE:?set DOC_VLLM_API_BASE (shared doc server)}"

MODEL="${MODEL:-qwen2.5-14b}"
DOC_MODEL="${DOC_MODEL:-qwen2.5-7b-docgen}"
GT_FILE="${GT_FILE:-$SCR/hotpot_dev_fullwiki_v1.json}"   # defaulted; override via env
VARIANT="${VARIANT:-search}"
MAX_Q="${MAX_Q:-50}"
NUM_RUNS="${NUM_RUNS:-10}"
CHARS_PER_DOC="${CHARS_PER_DOC:-500}"
MAX_TOKENS="${MAX_TOKENS:-512}"
DOC_MAX_TOKENS="${DOC_MAX_TOKENS:-512}"
SEED="${SEED:-42}"
DISTRACTOR_MODE="${DISTRACTOR_MODE:-rewrite}"
FRACTIONS="${FRACTIONS:-0 0.3 0.5 0.7}"   # 0 = matched no-distractor baseline (always include it)
CACHE_DIR="${CACHE_DIR:-$SCR/hf_cache}"
INDEX_DIR="${INDEX_DIR:-$SCR/hotpotqa_index}"
OUTDIR="${OUTDIR:-/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/base_hotpotqa_distractors/$MODEL}"
mkdir -p "$OUTDIR"

NATIVE_ARG=""
[[ "$DISTRACTOR_MODE" == "native_noise" ]] && NATIVE_ARG="--native-hotpot-file ${NATIVE_FILE:-$GT_FILE}"

COMMON=(--vllm-api-base "$VLLM_API_BASE" --model-name "$MODEL"
        --doc-vllm-api-base "$DOC_VLLM_API_BASE" --doc-model-name "$DOC_MODEL"
        --cache-dir "$CACHE_DIR" --index-dir "$INDEX_DIR"
        --pipeline-variant "$VARIANT" --max-questions "$MAX_Q" --seed "$SEED"
        --num-runs "$NUM_RUNS" --chars-per-doc "$CHARS_PER_DOC"
        --max-tokens "$MAX_TOKENS" --doc-max-tokens "$DOC_MAX_TOKENS"
        --doc-synthesis-mode faithful)

OUTS=()
for f in $FRACTIONS; do
  out="$OUTDIR/base_${VARIANT}_f${f}.json"   # non-overwriting: fraction in the filename
  echo "########## fraction=$f -> $out ##########"
  case "$f" in
    0|0.0|"")   # matched no-distractor baseline: faithful base loop, no distractor flags (byte-identical)
      python -u hotpot_pipeline.py "${COMMON[@]}" --output-path "$out"
      ;;
    *)          # distractor arm: faithful synthesis + round-0 DIVERSE distractors
      python -u hotpot_pipeline.py "${COMMON[@]}" \
        --distractor-fraction "$f" --distractor-mode "$DISTRACTOR_MODE" \
        --gt-file "$GT_FILE" $NATIVE_ARG --output-path "$out"
      ;;
  esac
  OUTS+=("$out")
done

echo "########## comparison ##########"
python -u base_hotpotqa_distractors/compare_sweep.py \
  --gt-file "$GT_FILE" --summary "$OUTDIR/sweep_summary_${VARIANT}.json" "${OUTS[@]}"
echo "Done. Per-fraction outputs + sweep_summary_${VARIANT}.json in $OUTDIR/"
