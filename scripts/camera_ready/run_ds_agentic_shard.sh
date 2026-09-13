#!/bin/bash
#SBATCH -J ds-agentic-shard
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint="vram40|vram48|vram80"
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=120g
#SBATCH -t 48:00:00
#SBATCH -o ds_agentic_shard_%j.out
set -eo pipefail
source /modules/opt/linux-ubuntu24.04-x86_64/miniforge3/24.7.1/etc/profile.d/conda.sh
conda activate ragenv
export HF_HOME=/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hf_cache
# VLLM_USE_DEEP_GEMM=0 avoids the FP8 DeepGEMM kernel path on Hopper GPUs (uri-gpu009 etc.), where
# `deep_gemm` is not installed and the engine crashes at init. This bf16 7B does not need FP8 kernels.
export HF_HUB_CACHE="$HF_HOME"; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
cd ~/RAG-Collapse

: "${SHARD:?pass SHARD=NN via --export=ALL,SHARD=NN}"
DATASET=$HOME/ds_agentic_shard_${SHARD}.jsonl
OUT=$HOME/deepseek_agentic_rerun_shard${SHARD}.json
test -f "$DATASET" || { echo "missing dataset $DATASET"; exit 1; }

# Unique ports per job so parallel shards on one node never share 8000/8001. ServerLLM auto-discovers
# the served model from /v1/models, so a shared port silently latches a client onto a neighbor's server.
ANS_MODEL=deepseek-r1-distill-qwen-7b
DOC_MODEL=qwen2.5-7b
PA=$((20000 + SLURM_JOB_ID % 10000)); PB=$((PA + 1))

# --- answer server: DeepSeek with reasoning + tool calling, on GPU 0 ---
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --served-model-name $ANS_MODEL \
  --host 0.0.0.0 --port $PA --gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-seqs 128 --trust-remote-code \
  --reasoning-parser deepseek_r1 --enable-auto-tool-choice --tool-call-parser hermes \
  > ~/ds_shard_ans_$SLURM_JOB_ID.log 2>&1 &
A=$!
# --- doc-gen server: Qwen2.5-7B plain, on GPU 1 ---
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name $DOC_MODEL \
  --host 0.0.0.0 --port $PB --gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-seqs 128 \
  > ~/ds_shard_doc_$SLURM_JOB_ID.log 2>&1 &
B=$!
trap "kill $A $B 2>/dev/null || true" EXIT

# Wait until each server is up AND serving exactly the model we asked for (guards against port reuse).
wait_served () {  # $1=port $2=served-name $3=proc-pid $4=log
  for i in $(seq 1 180); do
    curl -sf http://localhost:$1/v1/models 2>/dev/null | grep -q "\"$2\"" && { echo "server $1 up serving $2"; return 0; }
    kill -0 $3 2>/dev/null || { echo "SERVER_DIED $2"; tail -30 "$4"; exit 1; }
    sleep 10
  done
  echo "SERVER_NOT_SERVING $2"; tail -30 "$4"; exit 1
}
wait_served $PA $ANS_MODEL $A ~/ds_shard_ans_$SLURM_JOB_ID.log
wait_served $PB $DOC_MODEL $B ~/ds_shard_doc_$SLURM_JOB_ID.log

python -u pipeline.py \
  --model-mode server --vllm-api-base http://localhost:$PA/v1 --model-name $ANS_MODEL \
  --max-tokens 4096 --temperature 0.7 \
  --doc-model-mode server --doc-vllm-api-base http://localhost:$PB/v1 --doc-model-name $DOC_MODEL --doc-max-tokens 512 \
  --pipeline-variant agentic_rag \
  --dataset-path "$DATASET" \
  --output-path "$OUT" \
  --num-runs 10 --search-top-k 10 --search-chunk-size 500 --search-chunk-overlap 50 --agentic-max-tool-calls 10
echo "### shard $SHARD done ###"
