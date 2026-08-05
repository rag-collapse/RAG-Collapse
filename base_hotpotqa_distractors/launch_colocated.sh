#!/bin/bash
# CO-LOCATED single-job launcher for the base-HotpotQA round-0 distractor sweeps.
#
# WHY: the original launch.sh starts the GPU vLLM servers, then submits the sweep client as a
# SEPARATE CPU job. While that client waits in the CPU queue (or if it dies), the GPUs sit idle —
# which trips Unity's "GPU idle > 1h" reclaim policy (it killed xm-main-qwen2.5-14b). This launcher
# instead starts BOTH vLLM servers AND runs the client INSIDE ONE 2-GPU job, so the GPUs are driven
# the moment they're allocated: no cross-queue gap, no orphaned servers, no reaper.
#
# One job = one answer model + one DISTRACTOR_MODE + all VARIANTS/FRACTIONS(or TOPICS). At 400q a
# single sweep fits well inside the 48h wall, so run diverse_synth and equal_diverse_synth as
# SEPARATE jobs (do NOT chain them).
#
# Submit (override -J and --constraint on the command line; everything else via --export):
#   sbatch -J colo-qwen-diverse --constraint="bf16&vram48" \
#     --export=ALL,ANSWER_MODEL_ID=Qwen/Qwen2.5-14B-Instruct,ANSWER_SERVED=qwen2.5-14b,\
# ANSWER_PORT=5200,DOCGEN_PORT=5201,DISTRACTOR_MODE=diverse_synth,TEMPERATURE=1.0,TOP_P=1.0,\
# MAX_Q=400,OUTDIR=$HOME/temp1_sweeps/diverse_synth/qwen2.5-14b \
#     base_hotpotqa_distractors/launch_colocated.sh
# (submit_temp1_sweeps.sh wraps all 8 of these.)
#
#SBATCH -J colo-distractor
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=bf16&vram48
#SBATCH -c 12
#SBATCH --mem=112g
#SBATCH -t 48:00:00
#SBATCH -o logs/colo_%j.out
#SBATCH -e logs/colo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR" || exit 1; fi

module load conda/latest
module load cuda/12.6
conda activate ragenv

SCR=/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu
export HF_HOME="${HF_HOME:-$SCR/hf_cache}"
export HF_HUB_CACHE="$HF_HOME"
export GLOO_SOCKET_IFNAME=lo
export NCCL_DEBUG=ERROR
# Robust to GPU placement (skip DeepGEMM FP8 warmup that aborts on some Hopper GPUs).
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
mkdir -p logs

# ---- answer + doc server config (both bind localhost; the job's 2 GPUs are indices 0 and 1) ----
ANSWER_MODEL_ID="${ANSWER_MODEL_ID:-Qwen/Qwen2.5-14B-Instruct}"
ANSWER_SERVED="${ANSWER_SERVED:-qwen2.5-14b}"
ANSWER_EXTRA_ARGS="${ANSWER_EXTRA_ARGS:-}"        # Mistral: "--tokenizer-mode mistral"; DeepSeek: EMPTY
ANSWER_MAX_NUM_SEQS="${ANSWER_MAX_NUM_SEQS:-64}"  # 64 for 14B, 128 for 7-8B
ANSWER_PORT="${ANSWER_PORT:-5200}"
DOCGEN_PORT="${DOCGEN_PORT:-5201}"
DOC_MODEL_ID="${DOC_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
DOC_SERVED="${DOC_SERVED:-qwen2.5-7b-docgen}"
# Cap context length. Some answer models are 128K-native (Llama-3.1-8B, DeepSeek-R1-Distill-Qwen-7B);
# at 131072 the KV cache can't fit even one sequence on a <=24GB GPU and vLLM's engine init FAILS
# ("Available KV cache memory: 3.1 GiB"). HotpotQA prompts are short (question + ~10 short docs +
# a <=512-token answer), so 8192 is ample and lets these models fit the bf16&vram23 pool. 32K-native
# models (Mistral-7B, Qwen2.5-7B/14B) are unaffected by the cap.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
LOAD_TRIES="${LOAD_TRIES:-180}"   # 180 * 10s = 30 min max for a server to load its weights

alog="logs/colo_${SLURM_JOB_ID}_answer.log"
dlog="logs/colo_${SLURM_JOB_ID}_docgen.log"

echo "### co-located job ${SLURM_JOB_ID} on $(hostname -s): answer=$ANSWER_SERVED (GPU0 :$ANSWER_PORT), doc=$DOC_SERVED (GPU1 :$DOCGEN_PORT) ###"

CUDA_VISIBLE_DEVICES=0 vllm serve "$ANSWER_MODEL_ID" \
  --host 127.0.0.1 --port "$ANSWER_PORT" --served-model-name "$ANSWER_SERVED" \
  --tensor-parallel-size 1 --max-num-seqs "$ANSWER_MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 \
  --trust-remote-code $ANSWER_EXTRA_ARGS > "$alog" 2>&1 &
A_PID=$!

CUDA_VISIBLE_DEVICES=1 vllm serve "$DOC_MODEL_ID" \
  --host 127.0.0.1 --port "$DOCGEN_PORT" --served-model-name "$DOC_SERVED" \
  --tensor-parallel-size 1 --max-num-seqs 128 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
  --trust-remote-code > "$dlog" 2>&1 &
D_PID=$!

# Servers die when the job ends anyway (cgroup cleanup); this also stops them on early client exit.
cleanup() { echo "### cleanup: stopping servers ($A_PID $D_PID) ###"; kill "$A_PID" "$D_PID" 2>/dev/null || true; }
trap cleanup EXIT

wait_serve () {   # port pid name log
  local port="$1" pid="$2" name="$3" log="$4"
  echo "### waiting for $name on :$port (up to $((LOAD_TRIES*10/60)) min) ###"
  for _ in $(seq 1 "$LOAD_TRIES"); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "!!! $name process exited during startup — tail of $log:"; tail -n 30 "$log" 2>/dev/null; return 1
    fi
    if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then echo "  $name ready on :$port"; return 0; fi
    sleep 10
  done
  echo "!!! $name never became ready — tail of $log:"; tail -n 30 "$log" 2>/dev/null; return 1
}
wait_serve "$ANSWER_PORT" "$A_PID" answer "$alog" || exit 1
wait_serve "$DOCGEN_PORT" "$D_PID" docgen "$dlog" || exit 1

# ---- run the sweep client INLINE (server mode, localhost). run_sweep.sh sets CUDA_VISIBLE_DEVICES=""
#      for itself (CPU embeddings); the servers already hold the GPUs. ----
export VLLM_API_BASE="http://127.0.0.1:$ANSWER_PORT/v1"
export DOC_VLLM_API_BASE="http://127.0.0.1:$DOCGEN_PORT/v1"
export MODEL="$ANSWER_SERVED"
export DOC_MODEL="$DOC_SERVED"
export DOC_MODEL_MODE=server
# All remaining knobs (DISTRACTOR_MODE, DISTRACTOR_MODEL[_MODE], LITELLM_API_BASE, INITIAL_DOCS,
# NATIVE_FILE, VARIANTS, FRACTIONS, TOPICS, DOCS_PER_TOPIC, NUM_ITERATIONS, MAX_Q, NUM_RUNS, SEED,
# TEMPERATURE, TOP_P, DISTRACTOR_AVOID_GOLD, OUTDIR) pass straight through the environment.
echo "### servers up — running the sweep client (mode=${DISTRACTOR_MODE:-?}, temp=${TEMPERATURE:-def}/top_p=${TOP_P:-def}, MAX_Q=${MAX_Q:-?}) ###"
bash base_hotpotqa_distractors/run_sweep.sh
echo "### sweep client finished — cleanup trap will stop the servers ###"
