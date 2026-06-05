#!/bin/bash
#SBATCH -J gepa-vllm-deepseek-r1-7b
#SBATCH -c 6
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40|vram48|vram80
#SBATCH -t 48:00:00
#SBATCH -o logs/slurm-%j-gepa-vllm-deepseek-r1-7b.out
#SBATCH -e logs/slurm-%j-gepa-vllm-deepseek-r1-7b-error.out
#SBATCH --mail-type=ALL

model_cache_dir="/scratch4/workspace/rsenapati_umass_edu-rag-collapse/hf_cache"

module load conda/latest
module load cuda/12.6
conda activate ragenv

export HF_HOME="${model_cache_dir}"
export HF_HUB_CACHE="${model_cache_dir}"
mkdir -p "${model_cache_dir}"

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
log_file=./logs/slurm-$SLURM_JOB_ID-gepa-vllm-${SERVED_MODEL_NAME}.log

mkdir -p logs

{
echo "Launching vLLM server for ${MODEL_NAME} (GEPA pipeline — no tool calling)"
echo "Hostname: $(hostname)"
echo "FQDN: ${FQDN}"
echo "Listening endpoint: http://${HOST}:${PORT}/v1"
echo "Reachable at:      http://${FQDN}:${PORT}/v1"
} | tee "${log_file}"

# DeepSeek-R1-Distill-Qwen uses the Qwen2.5 tokenizer.
# --reasoning-parser deepseek_r1 strips <think>...</think> tokens so the
# pipeline receives clean final answers without the internal reasoning trace.
# Tool calling flags are omitted — this server is for non-agentic RAG only.

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
  --reasoning-parser deepseek_r1 2>&1 | tee -a "${log_file}"
