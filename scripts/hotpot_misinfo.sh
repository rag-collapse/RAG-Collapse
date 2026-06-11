#!/bin/bash
# --- SLURM ---
#SBATCH --job-name=hotpot-misinfo
#SBATCH --output=logs/hotpot_misinfo_%A.out
#SBATCH --error=logs/hotpot_misinfo_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram48|vram80
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#
# Error-compounding experiment (misinformation document synthesis) on HotpotQA.
# Runs the matched 3-arm pilot — faithful control, counterfactual, freeform — over a
# fixed question cohort and seed, so arms differ only in factual correctness.
#
# Prerequisites (start TWO vLLM servers first, then export their URLs):
#   export VLLM_API_BASE="http://<answer-host>:5150/v1"        # answer model (e.g. Qwen2.5-14B)
#   export DOC_VLLM_API_BASE="http://<docgen-host>:5153/v1"    # doc generator (Qwen2.5-7B)
#   export GT_FILE=/path/to/hotpot_dev_fullwiki_v1.json        # gold answers + (for hop) supporting_facts
#   sbatch --export=ALL scripts/hotpot_misinfo.sh

set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR" || exit 1; fi

module load conda/latest
module load cuda/12.6
conda activate ragenv

CACHE_DIR="${CACHE_DIR:-/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache/}"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"
mkdir -p logs

: "${VLLM_API_BASE:?set VLLM_API_BASE to the answer-model vLLM server}"
: "${DOC_VLLM_API_BASE:?set DOC_VLLM_API_BASE to the doc-generation vLLM server}"
: "${GT_FILE:?set GT_FILE to the native HotpotQA JSON (gold answers)}"

MODEL="${MODEL:-qwen2.5-14b}"
DOC_MODEL="${DOC_MODEL:-qwen2.5-7b-docgen}"
VARIANT="${VARIANT:-search}"
TARGET="${TARGET:-final_answer}"        # final_answer | intermediate_hop | untargeted
MAX_Q="${MAX_Q:-50}"                     # pilot cohort size
SEED="${SEED:-42}"
INJECT_ROUND="${INJECT_ROUND:-1}"
OUTDIR="${OUTDIR:-hotpot_misinfo_outputs}"
mkdir -p "$OUTDIR"

# hop target needs the native supporting_facts file (defaults to GT_FILE, which is native)
NATIVE_ARG=""
if [[ "$TARGET" == "intermediate_hop" ]]; then
  NATIVE_ARG="--native-hotpot-file ${NATIVE_FILE:-$GT_FILE}"
fi

run_arm () {
  local mode="$1"; local extra="$2"
  local out="$OUTDIR/hotpot_${VARIANT}_${TARGET}_${mode}.json"
  echo "=== arm: $mode -> $out ==="
  python -u hotpot_pipeline.py \
    --vllm-api-base "$VLLM_API_BASE" --model-name "$MODEL" \
    --doc-vllm-api-base "$DOC_VLLM_API_BASE" --doc-model-name "$DOC_MODEL" \
    --pipeline-variant "$VARIANT" --max-questions "$MAX_Q" --seed "$SEED" \
    --doc-synthesis-mode "$mode" --target-mode "$TARGET" --inject-round "$INJECT_ROUND" \
    --output-path "$out" $extra
  # evaluate (injection metrics auto-activate when injection records are present)
  python -u hotpot_evaluation.py "$out" "${out%.json}_eval.json" \
    --gt-file "$GT_FILE" --summary-json "${out%.json}_summary.json" \
    --plot-file "${out%.json}_f1.png"
}

# control arm: identical settings, no injection (gt-file allowed but unused by faithful)
run_arm faithful ""
run_arm counterfactual "--gt-file $GT_FILE $NATIVE_ARG"
run_arm freeform "--gt-file $GT_FILE $NATIVE_ARG"

echo "Done. Outputs in $OUTDIR/"
