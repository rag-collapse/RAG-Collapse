#!/bin/bash
#SBATCH -J vllm-qwen2.5-14b
#SBATCH -c 6
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40|vram48|vram80
#SBATCH -t 48:00:00
#SBATCH -o outputs/slurm-%j-vllm-qwen2.5-14b.out
#SBATCH -e outputs/slurm-%j-vllm-qwen2.5-14b-error.out
#SBATCH --mail-type=ALL

model_cache_dir="/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache"

module load conda/latest
module load cuda/12.6
conda activate ragenv

# Point HuggingFace at the shared cache — vLLM reads HF_HOME automatically.
# If the model is already cached here, HF will not re-download it (ETag check).
export HF_HOME="${model_cache_dir}"
export HF_HUB_CACHE="${model_cache_dir}"
mkdir -p "${model_cache_dir}"

# Keep libraries quiet
export GLOO_SOCKET_IFNAME=lo
export NCCL_DEBUG=ERROR
export TORCH_CPP_LOG_LEVEL=ERROR
export GLOG_minloglevel=3

# If serving for document generation
# CHANGE TO Qwen/Qwen2.5-7B-Instruct
# Don't forget to change job name in the header as well

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
SERVED_MODEL_NAME="qwen2.5-7b"
PORT=5150
HOST="0.0.0.0"
TP_SIZE=1
GPU_UTIL=0.90
MAX_NUM_SEQS=128
NUM_BATCHED_TOKENS=8192

FQDN=$(hostname -f)
log_file=./logs/slurm-$SLURM_JOB_ID-vllm-${SERVED_MODEL_NAME}.log

{
echo "Launching vLLM server for ${MODEL_NAME}"
echo "Hostname: $(hostname)"
echo "FQDN: ${FQDN}"
echo "Listening endpoint: http://${HOST}:${PORT}/v1"
echo "Reachable at:      http://${FQDN}:${PORT}/v1"
} | tee "${log_file}"

# Start server in the foreground — Slurm keeps the allocation alive
# If you are using a reasoning model, you need to put the --reasoning-parser deepseek_r1 after the first line
# example
# vllm serve "${MODEL_NAME}" \
#   --reasoning-parser deepseek_r1 \
#   --host "${HOST}" \
# no need to change deepseek_r1 part, it seems to be universal

# Tool-call parser (required for agentic_rag variant):
#   Qwen2.5 family  → hermes         (current, matches tokenizer_config.json)
#   Mistral/Mixtral → mistral
#   Llama-3.x       → llama3_json
#   DeepSeek-V3/R1  → deepseek_v3   (also needs --reasoning-parser deepseek_r1 for R1)
# Remove these two flags if you are NOT using the agentic_rag pipeline variant.

vllm serve "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --uvicorn-log-level info \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes 2>&1 | tee -a "${log_file}"

# --disable-access-log-for-endpoints /health,/metrics \
# --disable-log-stats
# --disable-uvicorn-access-log   # re-add this to silence all request logs
