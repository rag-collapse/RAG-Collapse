#!/bin/bash
# --- SLURM ---
#SBATCH --job-name=hotpot_pipeline
#SBATCH --output=logs/hotpot_pipeline_%A.out
#SBATCH --error=logs/hotpot_pipeline_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -C sm_70|sm_75|sm_80|sm_86|sm_90
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL

# Run from submit dir so hotpot_pipeline.py and paths resolve
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
fi

# --- Conda ---
module load cuda/12.6
module load conda/latest
conda activate rag

VLLM_API_BASE=http://gpu028.unity.rc.umass.edu:5151/v1
DOC_VLLM_API_BASE=http://gpu032.unity.rc.umass.edu:5150/v1

# Ensure a valid cache dir for HF
CACHE_DIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hf_cache/"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"

if [[ -z "$VLLM_API_BASE" ]]; then
  echo "ERROR: VLLM_API_BASE is not set. Start the answer vLLM server first, then:"
  echo "  export VLLM_API_BASE=\"http://<fqdn>:5150/v1\""
  echo "  export DOC_VLLM_API_BASE=\"http://<fqdn>:5151/v1\""
  echo "  sbatch --export=ALL scripts/hotpot_pipeline.sh"
  exit 1
fi

if [[ -z "$DOC_VLLM_API_BASE" ]]; then
  echo "ERROR: DOC_VLLM_API_BASE is not set. Start the doc vLLM server first, then:"
  echo "  export DOC_VLLM_API_BASE=\"http://<fqdn>:5151/v1\""
  echo "  sbatch --export=ALL scripts/hotpot_pipeline.sh"
  exit 1
fi

echo "Using VLLM_API_BASE=$VLLM_API_BASE"
echo "Using DOC_VLLM_API_BASE=$DOC_VLLM_API_BASE"

# --- Config ---
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
DOC_MODEL="Qwen/Qwen2.5-7B-Instruct"
OUTDIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/experiment_outputs/hotpotqa/$MODEL"

COMMON="--model-name $MODEL --doc-model-name $DOC_MODEL --num-runs 10 --chars-per-doc 500"

mkdir -p logs "$OUTDIR"

run() {
  python -u hotpot_pipeline.py \
    --vllm-api-base "$VLLM_API_BASE" \
    --doc-vllm-api-base "$DOC_VLLM_API_BASE" \
    $COMMON "$@"
}

# --- Smoke test (1 question, 2 rounds) ---
# Uncomment the block below to quickly verify server connections and pipeline logic.
# Results go to experiment_outputs/smoke_test/ so they won't overwrite production outputs.
# Once confirmed working, comment this block out and uncomment the production runs below.
#
# mkdir -p experiment_outputs/smoke_test
# run --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 \
#   --num-iterations 5 --max-questions 10 \
#   --output-path experiment_outputs/smoke_test/hotpot_replace_all.json
#
# run --pipeline-variant replace_one --num-iterations 2 --max-questions 1 \
#   --output-path experiment_outputs/smoke_test/hotpot_replace_one.json
#
# run --pipeline-variant search --num-iterations 5 --max-questions 10 \
#   --output-path experiment_outputs/smoke_test/hotpot_search.json

# --- Production runs ---
# Run each variant separately for clean logs; comment/uncomment as needed.

# run --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 --max-questions 1400 --num-iterations 10 \
#  --output-path "$OUTDIR/hotpot_replace_all.json"

# run --pipeline-variant replace_one --max-questions 1400 --num-iterations 20 \
#  --output-path "$OUTDIR/hotpot_replace_one.json"

# run --pipeline-variant search --max-questions 1400 --num-iterations 30 \
#  --output-path "$OUTDIR/hotpot_search.json"