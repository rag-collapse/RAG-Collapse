#!/bin/bash
# vLLM server for the DOCUMENT-GENERATION model (HotpotQA misinfo experiment).
# Fixed to Qwen2.5-7B-Instruct across all arms; also used as the misinfo judge LLM.
#SBATCH -J vllm-docgen-qwen7b
#SBATCH -c 6
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
# Qwen2.5-7B (bf16, ~15GB) needs an Ampere+ GPU that supports bf16 and >=24GB. Require the `bf16`
# feature: every bf16-tagged node here is >=24GB (L4=24, L40S/A40=48, A100=40/80, H100) and modern,
# so this includes the abundant, idle L4 pool while EXCLUDING old GPUs that crash vLLM with
# "no kernel image is available" (M40 sm_52, V100 sm_70, RTX-8000/2080Ti Turing). This fixes both the
# (Priority) starvation (premium-only pool) and the cudaErrorNoKernelImageForDevice crash.
#SBATCH --constraint=bf16
#SBATCH -t 12:00:00
#SBATCH -o logs/slurm-%j-vllm-docgen.out
#SBATCH -e logs/slurm-%j-vllm-docgen-error.out
#SBATCH --mail-type=END,FAIL

model_cache_dir="/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hf_cache"

module load conda/latest
module load cuda/12.6
conda activate ragenv

export HF_HOME="${model_cache_dir}"
export HF_HUB_CACHE="${model_cache_dir}"
mkdir -p "${model_cache_dir}" logs

export GLOO_SOCKET_IFNAME=lo
export NCCL_DEBUG=ERROR

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-7b-docgen}"
PORT="${PORT:-5153}"

FQDN=$(hostname -f)
echo "Doc-gen server: http://${FQDN}:${PORT}/v1   (served name: ${SERVED_MODEL_NAME})"
echo "  -> export DOC_VLLM_API_BASE=\"http://${FQDN}:${PORT}/v1\""

vllm serve "${MODEL_NAME}" \
  --host 0.0.0.0 --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size 1 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
