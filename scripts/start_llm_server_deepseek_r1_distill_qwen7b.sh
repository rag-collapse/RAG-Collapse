#!/bin/bash
#SBATCH -J vllm-deepseek-r1-distill-qwen-7b-agentic-rag
#SBATCH -c 6
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40|vram48|vram80
#SBATCH -t 48:00:00
#SBATCH -o outputs/slurm-%j-vllm-deepseek-r1-distill-qwen-7b.out
#SBATCH -e outputs/slurm-%j-vllm-deepseek-r1-distill-qwen-7b-error.out
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

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SERVED_MODEL_NAME="deepseek-r1-distill-qwen-7b"
PORT=5152
HOST="0.0.0.0"
TP_SIZE=1
GPU_UTIL=0.90
MAX_NUM_SEQS=128
NUM_BATCHED_TOKENS=8192

FQDN=$(hostname -f)
export VLLM_API_BASE="http://${FQDN}:${PORT}/v1"
export DOC_VLLM_API_BASE="http://${FQDN}:${PORT}/v1"
log_file=./logs/slurm-$SLURM_JOB_ID-vllm-${SERVED_MODEL_NAME}.log

{
echo "Launching vLLM server for ${MODEL_NAME}"
echo "Hostname: $(hostname)"
echo "FQDN: ${FQDN}"
echo "Listening endpoint: http://${HOST}:${PORT}/v1"
echo "Reachable at:      http://${FQDN}:${PORT}/v1"
} | tee "${log_file}"

# Start server in the foreground — Slurm keeps the allocation alive
# DeepSeek-R1-Distill-Qwen uses the Qwen2.5 tokenizer, NOT DeepSeek-V3.
# --tool-call-parser must be hermes (not deepseek_v3) — using deepseek_v3 raises:
#   "DeepSeek-V3 Tool parser could not locate tool call start/end tokens"
# --reasoning-parser deepseek_r1 strips <think>...</think> tokens from responses.
# Remove --enable-auto-tool-choice and --tool-call-parser if NOT using agentic_rag.

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
  --reasoning-parser deepseek_r1 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes 2>&1 | tee -a "${log_file}"

# --disable-access-log-for-endpoints /health,/metrics \
# --disable-log-stats
# --disable-uvicorn-access-log   # re-add this to silence all request logs
