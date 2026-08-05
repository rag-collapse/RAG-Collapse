#!/bin/bash
# vLLM server for the ANSWER model (HotpotQA misinfo experiment). Model is parametrized via
# MODEL_NAME/SERVED_MODEL_NAME/EXTRA_VLLM_ARGS; the launcher overrides -J per model.
# Non-agentic search variant → no tool-calling flags needed.
#SBATCH -J vllm-answer
#SBATCH -c 6
#SBATCH --mem=48g
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram48|vram80
#SBATCH -t 12:00:00
#SBATCH -o logs/slurm-%j-vllm-answer.out
#SBATCH -e logs/slurm-%j-vllm-answer-error.out
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
# Disable vLLM's DeepGEMM FP8 warmup: on some Hopper GPUs it aborts startup with
# "DeepGEMM backend is not available or outdated" (deep_gemm not installed). Off = robust
# to GPU placement (only forgoes an FP8 speedup we don't rely on). Allow env override.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-14b}"
PORT="${PORT:-5154}"
# Model-specific vLLM flags (space-separated, intentionally word-split below). Examples:
#   Mistral-7B   -> EXTRA_VLLM_ARGS="--tokenizer-mode mistral"
#   DeepSeek-R1  -> EXTRA_VLLM_ARGS="--reasoning-parser deepseek_r1"  (keeps <think> out of message.content)
# No tool-call flags needed: search/replace_one/hybrid are NOT agentic.
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
# Match the canonical per-model servers: 64 for the 14B default (anti-OOM, see
# start_llm_server_qwen14b.sh), 128 for the 7-8B answer models (set by their launchers).
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

FQDN=$(hostname -f)
echo "Answer server: http://${FQDN}:${PORT}/v1   (served name: ${SERVED_MODEL_NAME})"
echo "  model: ${MODEL_NAME}   extra args: ${EXTRA_VLLM_ARGS:-<none>}"
echo "  -> export VLLM_API_BASE=\"http://${FQDN}:${PORT}/v1\""

vllm serve "${MODEL_NAME}" \
  --host 0.0.0.0 --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size 1 \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code ${EXTRA_VLLM_ARGS}
