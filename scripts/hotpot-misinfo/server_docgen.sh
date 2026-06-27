#!/bin/bash
# vLLM server for the DOCUMENT-GENERATION model (HotpotQA misinfo experiment).
# Fixed to Qwen2.5-7B-Instruct across all arms; also used as the misinfo judge LLM.
#SBATCH -J vllm-docgen-qwen7b
#SBATCH -c 6
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
# Qwen2.5-7B is small (~15GB bf16 weights); it fits on a 24GB GPU. Allow the cheaper, far less
# contended 24-32GB pools (l4/v100-32/m40) so this server doesn't starve on (Priority) waiting for
# a premium vram48+ GPU it doesn't need (the recurring doc-gen scheduling failure).
#SBATCH --constraint=vram24|vram32|vram40|vram48|vram80
#SBATCH -t 12:00:00
#SBATCH -o logs/slurm-%j-vllm-docgen.out
#SBATCH -e logs/slurm-%j-vllm-docgen-error.out
#SBATCH --mail-type=END,FAIL

model_cache_dir="/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache"

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
