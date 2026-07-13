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
# Answer-model sampling. Unset -> hotpot_pipeline.py defaults (0.7 / 0.9), so other sweeps are
# unchanged. The shuffled experiment sets these to 1.0 / 1.0 to maximize per-run answer diversity.
TEMPERATURE="${TEMPERATURE:-}"
TOP_P="${TOP_P:-}"
SEED="${SEED:-42}"
DISTRACTOR_MODE="${DISTRACTOR_MODE:-rewrite}"   # rewrite | substitution | native_noise | diverse_synth | equal_diverse_synth
FRACTIONS="${FRACTIONS:-0 0.3 0.5 0.7}"   # 0 = matched no-distractor baseline (always include it)
# equal_diverse_synth: sweep the NUMBER OF TOPICS instead of the fraction. T=0 is the matched
# no-distractor baseline; each topic gets DOCS_PER_TOPIC paragraphs (4 topics x 2 = the 8 non-gold slots).
TOPICS="${TOPICS:-0 1 2 3 4}"
DOCS_PER_TOPIC="${DOCS_PER_TOPIC:-2}"
VARIANTS="${VARIANTS:-$VARIANT}"          # space-separated; e.g. "search hybrid replace_one"
INITIAL_DOCS="${INITIAL_DOCS:-faiss}"     # faiss | native_distractor (2 gold + 8 distractors)
CACHE_DIR="${CACHE_DIR:-$SCR/hf_cache}"
INDEX_DIR="${INDEX_DIR:-$SCR/hotpotqa_index}"
OUTDIR="${OUTDIR:-/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/base_hotpotqa_distractors/$MODEL}"
mkdir -p "$OUTDIR"

# --- model backends ---
# Per-round AI-doc synthesis uses doc_llm (two-server style: the ANSWER model, server mode on the
# answer server). Round-0 distractors may use a SEPARATE strong model via DISTRACTOR_MODEL
# (e.g. azure/gpt-5-mini through keymaker), leaving the per-round loop on the answer model.
DOC_ARGS=(--doc-model-mode "$DOC_MODEL_MODE" --doc-model-name "$DOC_MODEL")
if [[ "$DOC_MODEL_MODE" == "server" ]]; then
  : "${DOC_VLLM_API_BASE:?set DOC_VLLM_API_BASE for DOC_MODEL_MODE=server}"
  DOC_ARGS+=(--doc-vllm-api-base "$DOC_VLLM_API_BASE")
fi

DIST_ARGS=(); DIST_MODE=""
if [[ -n "${DISTRACTOR_MODEL:-}" ]]; then
  DIST_MODE="${DISTRACTOR_MODEL_MODE:-api}"
  DIST_ARGS=(--distractor-model-mode "$DIST_MODE" --distractor-model-name "$DISTRACTOR_MODEL")
  [[ "$DIST_MODE" == "server" ]] && DIST_ARGS+=(--distractor-vllm-api-base "${DISTRACTOR_VLLM_API_BASE:?set DISTRACTOR_VLLM_API_BASE for distractor server mode}")
  # Reasoning distractor models (gpt-5*) need a generous budget or return EMPTY docs. Give the
  # DISTRACTOR model its own budget so the per-round doc model keeps its (smaller) doc_max_tokens.
  [[ "$DIST_MODE" == "api" ]] && DISTRACTOR_MAX_TOKENS="${DISTRACTOR_MAX_TOKENS:-2048}"
  [[ -n "${DISTRACTOR_MAX_TOKENS:-}" ]] && DIST_ARGS+=(--distractor-max-tokens "$DISTRACTOR_MAX_TOKENS")
fi

# Per-round doc budget. Only bump when the DOC model itself is a reasoning api model; the distractor
# model has its own --distractor-max-tokens above, so this stays small (512) for a server doc model.
if [[ "$DOC_MODEL_MODE" == "api" ]]; then
  DOC_MAX_TOKENS="${DOC_MAX_TOKENS:-2048}"
else
  DOC_MAX_TOKENS="${DOC_MAX_TOKENS:-512}"
fi

# Load API keys from .env when an api backend is used (sbatch --export=ALL won't carry an unexported
# var; mirrors ProprietaryLLM's load_dotenv()). LITELLM_API_BASE=openai -> OpenAI direct (OPENAI_API_KEY,
# avoids Azure's content filter); otherwise the keymaker proxy (API_KEY).
if [[ "$DOC_MODEL_MODE" == "api" || "$DIST_MODE" == "api" ]]; then
  for envf in "${SLURM_SUBMIT_DIR:-.}/.env" ./.env; do
    if [[ -f "$envf" ]]; then set -a; . "$envf"; set +a; echo "[env] loaded keys from $envf"; break; fi
  done
  if [[ "${LITELLM_API_BASE:-}" == "openai" ]]; then
    : "${OPENAI_API_KEY:?set OPENAI_API_KEY (OpenAI direct) in the environment or .env}"
  else
    : "${API_KEY:?set API_KEY (keymaker) in the environment or .env}"
  fi
fi

# Round-0 source: native_distractor seeds the 2-gold/8-distractor HotpotQA context (applies to ALL
# arms, including the fraction-0 baseline). Needs the distractor-setting file.
INITIAL_ARGS=()
if [[ "$INITIAL_DOCS" == "native_distractor" ]]; then
  : "${NATIVE_FILE:?set NATIVE_FILE (hotpot_dev_distractor_v1.json) for native_distractor seeding}"
  INITIAL_ARGS=(--initial-docs native_distractor --native-hotpot-file "$NATIVE_FILE")
fi

NATIVE_ARG=""
[[ "$DISTRACTOR_MODE" == "native_noise" ]] && NATIVE_ARG="--native-hotpot-file ${NATIVE_FILE:-$GT_FILE}"

# Option A: give each run a different single distractor (wide round-0 answer distribution).
PER_RUN_ARG=""
case "${DISTRACTOR_PER_RUN:-}" in 1|true|yes) PER_RUN_ARG="--distractor-per-run" ;; esac

# Never corrupt gold docs (inject only into non-gold slots, random/seeded).
AVOID_GOLD_ARG=""
case "${DISTRACTOR_AVOID_GOLD:-}" in 1|true|yes) AVOID_GOLD_ARG="--distractor-avoid-gold" ;; esac

# shuffled-diverse-synth: each of the num_runs parallel generations gets a different seeded
# context-doc order (re-shuffled per round). Applied to ALL arms incl. the f0 baseline.
SHUFFLE_ARG=""
case "${SHUFFLE_PER_RUN:-}" in 1|true|yes) SHUFFLE_ARG="--shuffle-per-run" ;; esac

for VAR in $VARIANTS; do
  echo "==================== variant=$VAR ===================="
  # "replace_all" is a display alias for the hybrid variant (all-synth / no-DB docs), matching the
  # baseline's naming; both run --pipeline-variant hybrid. Output filenames keep $VAR, so a
  # replace_all run writes base_replace_all_*.json (hybrid still works for older callers).
  PV="$VAR"; SYNTH_ARGS=()
  if [[ "$VAR" == "hybrid" || "$VAR" == "replace_all" ]]; then
    PV="hybrid"; SYNTH_ARGS=(--num-synth-docs 10 --num-db-docs 0)
  fi
  VARIANT_ARGS=(--pipeline-variant "$PV" "${SYNTH_ARGS[@]}")

  COMMON=(--vllm-api-base "$VLLM_API_BASE" --model-name "$MODEL"
          "${DOC_ARGS[@]}" "${DIST_ARGS[@]}" "${INITIAL_ARGS[@]}"
          --cache-dir "$CACHE_DIR" --index-dir "$INDEX_DIR"
          "${VARIANT_ARGS[@]}" --max-questions "$MAX_Q" --seed "$SEED"
          --num-runs "$NUM_RUNS" --chars-per-doc "$CHARS_PER_DOC"
          --max-tokens "$MAX_TOKENS" --doc-max-tokens "$DOC_MAX_TOKENS"
          --doc-synthesis-mode faithful)
  [[ -n "$SHUFFLE_ARG" ]] && COMMON+=("$SHUFFLE_ARG")
  # Answer-model sampling overrides (the doc model inherits --temperature unless --doc-temperature set).
  [[ -n "$TEMPERATURE" ]] && COMMON+=(--temperature "$TEMPERATURE")
  [[ -n "$TOP_P" ]] && COMMON+=(--top-p "$TOP_P")
  # NUM_ITERATIONS overrides the variant's default round count (e.g. 3 for this experiment).
  [[ -n "${NUM_ITERATIONS:-}" ]] && COMMON+=(--num-iterations "$NUM_ITERATIONS")

  OUTS=()
  if [[ "$DISTRACTOR_MODE" == "equal_diverse_synth" ]]; then
    # Topic sweep: T=0 baseline, then T distinct wrong answers x DOCS_PER_TOPIC paragraphs each.
    for T in $TOPICS; do
      out="$OUTDIR/base_${VAR}_t${T}.json"   # non-overwriting: variant + topic count in the filename
      echo "########## variant=$VAR topics=$T -> $out ##########"
      case "$T" in
        0|"")     # matched no-distractor baseline (no distractor flags)
          python -u hotpot_pipeline.py "${COMMON[@]}" --output-path "$out"
          ;;
        *)        # equal_diverse_synth arm: T topics, DOCS_PER_TOPIC paragraphs each
          python -u hotpot_pipeline.py "${COMMON[@]}" \
            --distractor-mode equal_diverse_synth \
            --distractor-num-topics "$T" --distractor-docs-per-topic "$DOCS_PER_TOPIC" \
            --gt-file "$GT_FILE" $NATIVE_ARG $PER_RUN_ARG $AVOID_GOLD_ARG --output-path "$out"
          ;;
      esac
      OUTS+=("$out")
    done
  else
    for f in $FRACTIONS; do
      out="$OUTDIR/base_${VAR}_f${f}.json"   # non-overwriting: variant + fraction in the filename
      echo "########## variant=$VAR fraction=$f -> $out ##########"
      case "$f" in
        0|0.0|"")   # matched no-distractor baseline (no distractor flags)
          python -u hotpot_pipeline.py "${COMMON[@]}" --output-path "$out"
          ;;
        *)          # distractor arm: round-0 distractors via DISTRACTOR_MODE
          python -u hotpot_pipeline.py "${COMMON[@]}" \
            --distractor-fraction "$f" --distractor-mode "$DISTRACTOR_MODE" \
            --gt-file "$GT_FILE" $NATIVE_ARG $PER_RUN_ARG $AVOID_GOLD_ARG --output-path "$out"
          ;;
      esac
      OUTS+=("$out")
    done
  fi

  if [[ -n "${SKIP_COMPARE:-}" ]]; then
    echo "########## SKIP_COMPARE set — skipping in-job compare for variant=$VAR (aggregate post-hoc) ##########"
  else
    echo "########## comparison (variant=$VAR) ##########"
    python -u base_hotpotqa_distractors/compare_sweep.py \
      --gt-file "$GT_FILE" --summary "$OUTDIR/sweep_summary_${VAR}.json" "${OUTS[@]}"
  fi
done
echo "Done. Per-variant sweep_summary_*.json + base_<variant>_f*.json in $OUTDIR/"
