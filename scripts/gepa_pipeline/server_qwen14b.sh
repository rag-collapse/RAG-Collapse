#!/bin/bash
#SBATCH -J gepa-vllm-qwen2.5-14b
#SBATCH -c 6
#SBATCH --mem=48g
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram48|vram80
#SBATCH -t 48:00:00
#SBATCH -o logs/slurm-%j-gepa-vllm-qwen2.5-14b.out
#SBATCH -e logs/slurm-%j-gepa-vllm-qwen2.5-14b-error.out
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

MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
SERVED_MODEL_NAME="qwen2.5-14b"
PORT=5154
HOST="0.0.0.0"
TP_SIZE=1
GPU_UTIL=0.92
MAX_NUM_SEQS=64          # lower than 7B models to avoid OOM on 48GB card
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

# Qwen2.5-14B fits on a single 48GB GPU (A6000/A100).
# Tool calling flags are omitted — this server is for non-agentic RAG only.
# Reduce MAX_NUM_SEQS or GPU_UTIL if you see CUDA OOM at startup.

vllm serve "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --uvicorn-log-level info \
  --trust-remote-code 2>&1 | tee -a "${log_file}"
