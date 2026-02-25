#!/bin/bash
#SBATCH -J vllm-qwen2.5-14b
#SBATCH -c 6
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:2
#SBATCH --constraint=vram40|vram48|vram80
#SBATCH -t 2:00:00
#SBATCH -o outputs/slurm-%j-vllm-qwen2.5-14b.out
#SBATCH -e outputs/slurm-%j-vllm-qwen2.5-14b-error.out
#SBATCH --mail-type=ALL

model_cache_dir="/work/pi_dagarwal_umass_edu/project_4/hf_cache"

module load conda/latest
module load cuda/12.6
conda activate ragenv

# Keep libraries quiet
export OMP_NUM_THREADS=6
export NCCL_ASYNC_ERROR_HANDLING=1

# --- Silence / avoid the bind + chatter on single node ---
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_DEBUG=ERROR
export TORCH_CPP_LOG_LEVEL=ERROR
export GLOG_minloglevel=3

MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"
SERVED_MODEL_NAME="qwen2.5-14b"
PORT=5150
HOST="0.0.0.0"
TP_SIZE=2
GPU_UTIL=0.90
MAX_NUM_SEQS=32
DOWNLOAD_DIR="${model_cache_dir}"

mkdir -p "${DOWNLOAD_DIR}"

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
vllm serve "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --uvicorn-log-level error \
  --disable-uvicorn-access-log \
  --disable-log-stats \
  --download-dir "${DOWNLOAD_DIR}" \
  --trust-remote-code 2>&1 | tee -a "${log_file}"Then to actually read requests:
