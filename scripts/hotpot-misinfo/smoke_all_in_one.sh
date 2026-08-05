#!/bin/bash
# All-in-one QUICK smoke: one GPU, one Qwen2.5-7B vLLM server used for BOTH the answer
# and doc-gen/judge roles, then the CPU smoke client + eval + self-checks in the same job.
#
# A smoke validates plumbing (injection records, metrics, eval) — not ASR magnitude — so a
# single shared model is fine, and one small GPU schedules far faster than two big servers
# with no cross-node URL coordination to get wrong. The server is torn down on exit.
#
# Run it (no manual server setup, no URL exports needed):
#   sbatch --export=ALL,MAX_Q=20 scripts/hotpot-misinfo/smoke_all_in_one.sh
# MAX_Q defaults to 5 in smoke_test.sh; a weak 7B answerer needs ~20 questions to reliably
# land at least one realized (status=ok) counterfactual injection. See RUNBOOK.md.
#
#SBATCH -J hotpot-misinfo-aio
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40|vram48|vram80
#SBATCH -c 8
#SBATCH --mem=96g
#SBATCH -t 02:00:00
#SBATCH -o logs/hotpot_misinfo_aio_%j.out
#SBATCH -e logs/hotpot_misinfo_aio_%j.err
#SBATCH --mail-type=END,FAIL
set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR" || exit 1; fi

module load conda/latest
module load cuda/12.6
conda activate ragenv

SCR=/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu
export HF_HOME="$SCR/hf_cache"
export HF_HUB_CACHE="$HF_HOME"
export GLOO_SOCKET_IFNAME=lo
export NCCL_DEBUG=ERROR
mkdir -p logs

PORT="${PORT:-5153}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
SERVED="${SERVED:-qwen2.5-7b}"
VLOG="logs/aio_vllm_${SLURM_JOB_ID}.out"

echo "### starting vllm $MODEL_ID on port $PORT ###"
vllm serve "$MODEL_ID" \
  --host 0.0.0.0 --port "$PORT" --served-model-name "$SERVED" \
  --tensor-parallel-size 1 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --trust-remote-code > "$VLOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

echo "### waiting for server readiness (up to ~20 min) ###"
UP=0
for i in $(seq 1 120); do
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then UP=1; echo "server up after $((i*10))s"; break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "!!! vllm process died — last 60 log lines:"; tail -60 "$VLOG"; exit 1; fi
  sleep 10
done
if [[ $UP -ne 1 ]]; then echo "!!! server never became ready — last 60 log lines:"; tail -60 "$VLOG"; exit 1; fi

export VLLM_API_BASE="http://localhost:$PORT/v1"
export DOC_VLLM_API_BASE="http://localhost:$PORT/v1"
export MODEL="$SERVED"
export DOC_MODEL="$SERVED"

echo "### running smoke client (3 arms + eval + checks) ###"
bash scripts/hotpot-misinfo/smoke_test.sh
RC=$?
echo "### smoke finished rc=$RC ###"
exit $RC
