#!/bin/bash
# Build the frozen, reproducible distractor-document pools ONCE (CPU + API only — NO GPU).
#
# The round-0 distractor docs depend only on (question, gold, seed, mode) — not on the answer model
# or the variant — and are synthesized by an expensive, non-reproducible API model (gpt-5-mini). So
# we synthesize them once at the FULL pool (all 8 non-gold native slots) and freeze them to a JSON
# dataset; every experiment arm then loads the pool (--distractor-docs-file) and applies a nested
# prefix per fraction/topic (no API calls in the runs). This both slashes API cost (2 builds vs ~108
# arms) and makes the distractors a fixed, releasable artifact for the paper.
#
# Runs as a CPU batch job (no GPU held, so no idle-reclaim concern). Needs OPENAI_API_KEY in ./.env.
#   sbatch base_hotpotqa_distractors/build_distractor_pools.sh                 # full 1400q, both modes
#   MAX_Q=5 sbatch base_hotpotqa_distractors/build_distractor_pools.sh         # smoke
#
#SBATCH -J build-distractor-pools
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -t 12:00:00
#SBATCH -o logs/build_pools_%j.out
#SBATCH -e logs/build_pools_%j.err
#SBATCH --mail-type=END,FAIL
set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR" || exit 1; fi

module load conda/latest
conda activate ragenv

SCR=/scratch4/workspace/oyilmazel_umass_edu-rag_collapse
export HF_HOME="${HF_HOME:-$SCR/hf_cache}"
export HF_HUB_CACHE="$HF_HOME"
export CUDA_VISIBLE_DEVICES=""      # CPU + API only; the answer/doc GPU servers are never built in dump mode
mkdir -p logs

# Load OPENAI_API_KEY (OpenAI-direct avoids Azure's content filter); mirrors run_sweep.sh.
for envf in "${SLURM_SUBMIT_DIR:-.}/.env" ./.env; do
  if [[ -f "$envf" ]]; then set -a; . "$envf"; set +a; echo "[env] loaded keys from $envf"; break; fi
done
export LITELLM_API_BASE="${LITELLM_API_BASE:-openai}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (OpenAI direct) in the environment or ./.env}"

NATIVE="${NATIVE_FILE:-$SCR/hotpot_dev_distractor_v1.json}"
GT="${GT_FILE:-$SCR/hotpot_dev_fullwiki_v1.json}"
CACHE="${CACHE_DIR:-$SCR/hf_cache}"
INDEX="${INDEX_DIR:-$SCR/hotpotqa_index}"
MAXQ="${MAX_Q:-1400}"
SEED="${SEED:-42}"
DISTMODEL="${DISTRACTOR_MODEL:-gpt-5-mini}"
POOL_DIR="${POOL_DIR:-$HOME/distractor_pools}"
mkdir -p "$POOL_DIR"

# replace_one keeps all 10 native docs at round 0, so corrupting the non-gold slots yields the FULL
# 8-doc pool that every variant (search / replace_one / replace_all) draws its nested prefix from.
common=(--model-name qwen2.5-14b --vllm-api-base "http://127.0.0.1:1/v1"
        --doc-model-mode api --doc-model-name "$DISTMODEL"
        --distractor-model-mode api --distractor-model-name "$DISTMODEL" --distractor-max-tokens 2048
        --initial-docs native_distractor --native-hotpot-file "$NATIVE"
        --pipeline-variant replace_one --distractor-avoid-gold
        --gt-file "$GT" --max-questions "$MAXQ" --seed "$SEED"
        --cache-dir "$CACHE" --index-dir "$INDEX")

echo "### [1/2] diverse_synth pool (full: --distractor-fraction 1.0) -> $POOL_DIR/diverse_synth_pool.json ###"
python -u hotpot_pipeline.py "${common[@]}" \
  --distractor-mode diverse_synth --distractor-fraction 1.0 \
  --dump-distractor-docs "$POOL_DIR/diverse_synth_pool.json"

echo "### [2/2] equal_diverse_synth pool (full: 4 topics x 2 docs) -> $POOL_DIR/equal_diverse_synth_pool.json ###"
python -u hotpot_pipeline.py "${common[@]}" \
  --distractor-mode equal_diverse_synth --distractor-num-topics 4 --distractor-docs-per-topic 2 \
  --dump-distractor-docs "$POOL_DIR/equal_diverse_synth_pool.json"

echo "### done. pools in $POOL_DIR: ###"
ls -la "$POOL_DIR"