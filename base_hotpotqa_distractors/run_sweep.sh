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

MODEL="${MODEL:-qwen2.5-14b}"
# DOC_MODEL_MODE: 'server' (default, vLLM doc server) or 'api' (keymaker LiteLLM strong model).
# For api mode set DOC_MODEL to a keymaker id (e.g. openai/claude-sonnet-4-6) and export API_KEY.
DOC_MODEL_MODE="${DOC_MODEL_MODE:-server}"
DOC_MODEL="${DOC_MODEL:-qwen2.5-7b-docgen}"
GT_FILE="${GT_FILE:-$SCR/hotpot_dev_fullwiki_v1.json}"   # defaulted; override via env
VARIANT="${VARIANT:-search}"
MAX_Q="${MAX_Q:-50}"
NUM_RUNS="${NUM_RUNS:-10}"
CHARS_PER_DOC="${CHARS_PER_DOC:-500}"
MAX_TOKENS="${MAX_TOKENS:-512}"
# Reasoning API models (gpt-5*) spend the token budget on hidden reasoning tokens; 512 leaves the
# visible document EMPTY. Default the doc budget higher in api mode (override via DOC_MAX_TOKENS).
if [[ "$DOC_MODEL_MODE" == "api" ]]; then
  DOC_MAX_TOKENS="${DOC_MAX_TOKENS:-2048}"
else
  DOC_MAX_TOKENS="${DOC_MAX_TOKENS:-512}"
fi
SEED="${SEED:-42}"
DISTRACTOR_MODE="${DISTRACTOR_MODE:-rewrite}"   # rewrite | substitution | native_noise | diverse_synth
FRACTIONS="${FRACTIONS:-0 0.3 0.5 0.7}"   # 0 = matched no-distractor baseline (always include it)
CACHE_DIR="${CACHE_DIR:-$SCR/hf_cache}"
INDEX_DIR="${INDEX_DIR:-$SCR/hotpotqa_index}"
OUTDIR="${OUTDIR:-/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/base_hotpotqa_distractors/$MODEL}"
mkdir -p "$OUTDIR"

# Doc-generation backend: api (keymaker) needs API_KEY but no doc server; server needs DOC_VLLM_API_BASE.
DOC_ARGS=(--doc-model-mode "$DOC_MODEL_MODE" --doc-model-name "$DOC_MODEL")
if [[ "$DOC_MODEL_MODE" == "server" ]]; then
  : "${DOC_VLLM_API_BASE:?set DOC_VLLM_API_BASE (shared doc server) for DOC_MODEL_MODE=server}"
  DOC_ARGS+=(--doc-vllm-api-base "$DOC_VLLM_API_BASE")
else
  # The keymaker API_KEY may live in a .env file (repo root) rather than the exported
  # environment — sbatch --export=ALL won't carry an unexported var. Load it from .env if
  # not already set (mirrors ProprietaryLLM's load_dotenv()).
  if [[ -z "${API_KEY:-}" ]]; then
    for envf in "${SLURM_SUBMIT_DIR:-.}/.env" ./.env; do
      if [[ -f "$envf" ]]; then
        set -a; . "$envf"; set +a
        echo "[env] loaded API_KEY from $envf"
        break
      fi
    done
  fi
  : "${API_KEY:?set API_KEY (export it or put it in .env at the repo root) for DOC_MODEL_MODE=api}"
fi

NATIVE_ARG=""
[[ "$DISTRACTOR_MODE" == "native_noise" ]] && NATIVE_ARG="--native-hotpot-file ${NATIVE_FILE:-$GT_FILE}"

# Option A: give each run a different single distractor (wide round-0 answer distribution).
PER_RUN_ARG=""
case "${DISTRACTOR_PER_RUN:-}" in 1|true|yes) PER_RUN_ARG="--distractor-per-run" ;; esac

COMMON=(--vllm-api-base "$VLLM_API_BASE" --model-name "$MODEL"
        "${DOC_ARGS[@]}"
        --cache-dir "$CACHE_DIR" --index-dir "$INDEX_DIR"
        --pipeline-variant "$VARIANT" --max-questions "$MAX_Q" --seed "$SEED"
        --num-runs "$NUM_RUNS" --chars-per-doc "$CHARS_PER_DOC"
        --max-tokens "$MAX_TOKENS" --doc-max-tokens "$DOC_MAX_TOKENS"
        --doc-synthesis-mode faithful)
# NUM_ITERATIONS overrides the variant's default round count (e.g. 1 for a quick smoke).
[[ -n "${NUM_ITERATIONS:-}" ]] && COMMON+=(--num-iterations "$NUM_ITERATIONS")

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
        --gt-file "$GT_FILE" $NATIVE_ARG $PER_RUN_ARG --output-path "$out"
      ;;
  esac
  OUTS+=("$out")
done

echo "########## comparison ##########"
python -u base_hotpotqa_distractors/compare_sweep.py \
  --gt-file "$GT_FILE" --summary "$OUTDIR/sweep_summary_${VARIANT}.json" "${OUTS[@]}"
echo "Done. Per-fraction outputs + sweep_summary_${VARIANT}.json in $OUTDIR/"
