#!/bin/bash
# CPU client: the ORIGINAL HotpotQA paper distractor setting (Yang et al., EMNLP 2018) run
# through the recursive RAG-collapse loop. Round 0 is seeded from each question's native
# 2-gold + 8-TF-IDF-distractor context (NO FAISS), faithful synthesis, then the loop. Also runs
# a gold-only contrast (2 gold, no distractors). Talks to two SHARED vLLM servers; launched by
# launch_native.sh.
#SBATCH -J base-native-distractor
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 47:00:00
#SBATCH -o logs/base_native_%A.out
#SBATCH -e logs/base_native_%A.err
#SBATCH --mail-type=END,FAIL
set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR" || exit 1; fi

module load conda/latest
conda activate ragenv

SCR=/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu
export HF_HOME="${HF_HOME:-$SCR/hf_cache}"
export HF_HUB_CACHE="$HF_HOME"
export CUDA_VISIBLE_DEVICES=""        # server-mode client; E5 (search variant) runs on CPU
mkdir -p logs

: "${VLLM_API_BASE:?set VLLM_API_BASE (shared answer server)}"
: "${DOC_VLLM_API_BASE:?set DOC_VLLM_API_BASE (shared doc server)}"

MODEL="${MODEL:-qwen2.5-14b}"
DOC_MODEL="${DOC_MODEL:-qwen2.5-7b-docgen}"
# Paper distractor-setting file (2 gold + 8 distractors); also carries gold answers for eval.
NATIVE_FILE="${NATIVE_FILE:-$SCR/hotpot_dev_distractor_v1.json}"
GT_FILE="${GT_FILE:-$NATIVE_FILE}"
VARIANT="${VARIANT:-replace_one}"     # replace_one keeps all 10 native docs at round 0; see README for the search caveat
MAX_Q="${MAX_Q:-50}"
NUM_RUNS="${NUM_RUNS:-10}"
CHARS_PER_DOC="${CHARS_PER_DOC:-500}"
MAX_TOKENS="${MAX_TOKENS:-512}"
DOC_MAX_TOKENS="${DOC_MAX_TOKENS:-512}"
SEED="${SEED:-42}"
ARMS="${ARMS:-distractor goldonly}"   # distractor setting + gold-only contrast
CACHE_DIR="${CACHE_DIR:-$SCR/hf_cache}"
INDEX_DIR="${INDEX_DIR:-$SCR/hotpotqa_index}"
OUTDIR="${OUTDIR:-/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/base_hotpotqa_distractors/native_$MODEL}"
mkdir -p "$OUTDIR"

# Guard: the distractor-setting file must exist (the launcher downloads it on the login node).
if [[ ! -f "$NATIVE_FILE" ]]; then
  echo "Distractor-setting file missing: $NATIVE_FILE" >&2
  echo "Generate it on a login node (has internet + ragenv): python base_hotpotqa_distractors/fetch_distractor_file.py $NATIVE_FILE" >&2
  echo "  (or just run launch_native.sh, which generates it automatically)" >&2
  exit 1
fi

COMMON=(--vllm-api-base "$VLLM_API_BASE" --model-name "$MODEL"
        --doc-vllm-api-base "$DOC_VLLM_API_BASE" --doc-model-name "$DOC_MODEL"
        --cache-dir "$CACHE_DIR" --index-dir "$INDEX_DIR"
        --pipeline-variant "$VARIANT" --max-questions "$MAX_Q" --seed "$SEED"
        --num-runs "$NUM_RUNS" --chars-per-doc "$CHARS_PER_DOC"
        --max-tokens "$MAX_TOKENS" --doc-max-tokens "$DOC_MAX_TOKENS"
        --doc-synthesis-mode faithful
        --initial-docs native_distractor --native-hotpot-file "$NATIVE_FILE")

OUTS=()
for arm in $ARMS; do
  out="$OUTDIR/native_${VARIANT}_${arm}.json"
  echo "########## arm=$arm -> $out ##########"
  EXTRA=""
  [[ "$arm" == "goldonly" ]] && EXTRA="--distractor-gold-only"
  python -u hotpot_pipeline.py "${COMMON[@]}" $EXTRA --output-path "$out"
  python -u hotpot_evaluation.py "$out" "${out%.json}_eval.json" \
    --gt-file "$GT_FILE" --summary-json "${out%.json}_summary.json" --plot-file "${out%.json}_f1.png"
  OUTS+=("$out")
done

echo "########## comparison (distractor setting vs gold-only) ##########"
python -u base_hotpotqa_distractors/compare_native.py \
  --gt-file "$GT_FILE" --summary "$OUTDIR/native_summary_${VARIANT}.json" "${OUTS[@]}"
echo "Done. Outputs + native_summary_${VARIANT}.json in $OUTDIR/"
