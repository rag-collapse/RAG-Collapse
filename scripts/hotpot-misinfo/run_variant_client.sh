#!/bin/bash
# One VARIANT of the misinfo experiment (all 3 arms), as a CPU client that talks to two
# SHARED vLLM servers. Several of these run in parallel against the same server pair —
# the answer/doc-gen servers are stateless, so one pair serves replace_one + hybrid +
# search at once (vLLM queues concurrent requests). Launch via launch_all_variants.sh.
#
#SBATCH -J hotpot-misinfo-cli
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=96g
#SBATCH -t 47:00:00
#SBATCH -o logs/hotpot_misinfo_cli_%A.out
#SBATCH -e logs/hotpot_misinfo_cli_%A.err
#SBATCH --mail-type=END,FAIL
#
# 47h (not 48h): the shared servers start ~10-15 min earlier, so this guarantees the
# servers outlive the client. Per-arm output is written before the next arm starts, so a
# wall-clock kill only loses the in-flight arm. Override with --time at submit if needed.
#
# Required env (set by the launcher, or export yourself):
#   VLLM_API_BASE, DOC_VLLM_API_BASE  — the two shared servers
#   GT_FILE                           — native HotpotQA JSON (gold answers)
#   VARIANT                           — search | replace_one | hybrid
# Optional: MODEL DOC_MODEL TARGET MAX_Q SEED INJECT_ROUND ARMS NATIVE_FILE OUTDIR
set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR" || exit 1; fi

module load conda/latest
conda activate ragenv

SCR=/scratch4/workspace/oyilmazel_umass_edu-rag_collapse
export HF_HOME="${HF_HOME:-$SCR/hf_cache}"
export HF_HUB_CACHE="$HF_HOME"
export CUDA_VISIBLE_DEVICES=""        # force E5 onto CPU — this is a CPU client
mkdir -p logs

: "${VLLM_API_BASE:?set VLLM_API_BASE (shared answer server)}"
: "${DOC_VLLM_API_BASE:?set DOC_VLLM_API_BASE (shared doc-gen server)}"
: "${GT_FILE:?set GT_FILE to the native HotpotQA JSON}"

MODEL="${MODEL:-qwen2.5-14b}"
DOC_MODEL="${DOC_MODEL:-qwen2.5-7b-docgen}"
VARIANT="${VARIANT:-search}"
TARGET="${TARGET:-final_answer}"
MAX_Q="${MAX_Q:-50}"
SEED="${SEED:-42}"
INJECT_ROUND="${INJECT_ROUND:-1}"
# Answer token ceiling. 512 is plenty for the non-reasoning models; DeepSeek-R1 needs much
# more headroom for its <think> trace (set MAX_TOKENS=4096 in its launcher). DOC_MAX_TOKENS
# is pinned to 512 for ALL models so synthesized-document length stays a controlled constant.
MAX_TOKENS="${MAX_TOKENS:-512}"
DOC_MAX_TOKENS="${DOC_MAX_TOKENS:-512}"
# Match the canonical hotpot run (scripts/hotpot_pipeline.sh): --num-runs 10 --chars-per-doc 500.
# chars-per-doc especially matters — it sets how much of each doc feeds the RAG prompt per round.
NUM_RUNS="${NUM_RUNS:-10}"
CHARS_PER_DOC="${CHARS_PER_DOC:-500}"
ARMS="${ARMS:-faithful counterfactual freeform}"
# Round-0 initial-doc distractors (default off). When >0, each arm also seeds DIVERSE
# wrong-answer distractors into the round-0 retrieved docs (needs --gt-file; native_noise
# also needs the native file). Independent of the synthesis arm.
DISTRACTOR_FRACTION="${DISTRACTOR_FRACTION:-0}"
DISTRACTOR_MODE="${DISTRACTOR_MODE:-rewrite}"
CACHE_DIR="${CACHE_DIR:-$SCR/hf_cache}"
INDEX_DIR="${INDEX_DIR:-$SCR/hotpotqa_index}"
OUTDIR="${OUTDIR:-hotpot_misinfo_outputs}"
mkdir -p "$OUTDIR"

NATIVE_ARG=""
if [[ "$TARGET" == "intermediate_hop" || "$DISTRACTOR_MODE" == "native_noise" ]]; then
  NATIVE_ARG="--native-hotpot-file ${NATIVE_FILE:-$GT_FILE}"
fi

DISTRACTOR_ARG=""
case "$DISTRACTOR_FRACTION" in
  0|0.0|"") ;;
  *) DISTRACTOR_ARG="--distractor-fraction $DISTRACTOR_FRACTION --distractor-mode $DISTRACTOR_MODE" ;;
esac

run_arm () {
  local mode="$1"; local extra="$2"
  local out="$OUTDIR/hotpot_${VARIANT}_${TARGET}_${mode}.json"
  echo "=== variant=$VARIANT arm=$mode -> $out ==="
  python -u hotpot_pipeline.py \
    --vllm-api-base "$VLLM_API_BASE" --model-name "$MODEL" \
    --doc-vllm-api-base "$DOC_VLLM_API_BASE" --doc-model-name "$DOC_MODEL" \
    --cache-dir "$CACHE_DIR" --index-dir "$INDEX_DIR" \
    --pipeline-variant "$VARIANT" --max-questions "$MAX_Q" --seed "$SEED" \
    --num-runs "$NUM_RUNS" --chars-per-doc "$CHARS_PER_DOC" \
    --max-tokens "$MAX_TOKENS" --doc-max-tokens "$DOC_MAX_TOKENS" \
    --doc-synthesis-mode "$mode" --target-mode "$TARGET" --inject-round "$INJECT_ROUND" \
    --output-path "$out" $extra $DISTRACTOR_ARG
  python -u hotpot_evaluation.py "$out" "${out%.json}_eval.json" \
    --gt-file "$GT_FILE" --summary-json "${out%.json}_summary.json" \
    --plot-file "${out%.json}_f1.png"
}

for arm in $ARMS; do
  if [[ "$arm" == "faithful" && -z "$DISTRACTOR_ARG" ]]; then
    run_arm faithful ""                              # byte-identical baseline (no gt-file)
  else
    run_arm "$arm" "--gt-file $GT_FILE $NATIVE_ARG"  # gt-file needed for injection and/or distractors
  fi
done

echo "Done variant=$VARIANT (arms: $ARMS). Outputs in $OUTDIR/"
