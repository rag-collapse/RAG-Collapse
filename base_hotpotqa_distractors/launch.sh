#!/bin/bash
# One-command launcher for the base-HotpotQA + round-0 distractor sweep. Run on a LOGIN node:
#
#   bash base_hotpotqa_distractors/launch.sh                 # qwen2.5-14b, default sweep
#   ANSWER_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3 ANSWER_SERVED=mistral-7b \
#     ANSWER_EXTRA_ARGS="--tokenizer-mode mistral" ANSWER_MAX_NUM_SEQS=128 bash ...   # other models
#
# Starts a shared answer + Qwen2.5-7B doc server (own ports, so it coexists with the running
# graphite baseline on 5154/5153 and the misinfo launchers on 5164-5170), waits until both serve,
# submits the fraction-sweep CPU client (run_sweep.sh), and submits a reaper that scancels the
# servers when the client finishes. Reuses the validated servers in scripts/hotpot-misinfo/.
set -eo pipefail
cd "$(dirname "$0")/.." || exit 1          # repo root (this script lives in base_hotpotqa_distractors/)

SERVER_TIME="${SERVER_TIME:-48:00:00}"
CLIENT_TIME="${CLIENT_TIME:-47:00:00}"
# How long to wait for EACH server to become ready before giving up. Under heavy GPU contention a
# second GPU can take a long time to schedule, so default generously (WAIT_TRIES*WAIT_SLEEP seconds).
WAIT_TRIES="${WAIT_TRIES:-900}"     # 900 * 20s = 5h
WAIT_SLEEP="${WAIT_SLEEP:-20}"
ANSWER_PORT="${ANSWER_PORT:-5180}"
DOCGEN_PORT="${DOCGEN_PORT:-5181}"
ANSWER_MODEL_ID="${ANSWER_MODEL_ID:-Qwen/Qwen2.5-14B-Instruct}"
ANSWER_SERVED="${ANSWER_SERVED:-qwen2.5-14b}"
ANSWER_EXTRA_ARGS="${ANSWER_EXTRA_ARGS:-}"      # e.g. "--tokenizer-mode mistral"; DeepSeek: leave empty
ANSWER_MAX_NUM_SEQS="${ANSWER_MAX_NUM_SEQS:-64}"  # 64 for 14B (anti-OOM); set 128 for 7-8B models
# Doc-generation backend. 'server' (default) starts a Qwen2.5-7B doc GPU server; 'api' routes doc
# generation through keymaker (a strong LiteLLM model) and SKIPS the doc GPU server entirely.
# For api mode: set DOC_MODEL to a keymaker id (e.g. openai/claude-sonnet-4-6) and put your API_KEY
# in .env at the repo root (run_sweep.sh loads it) or export it before launching.
DOC_MODEL_MODE="${DOC_MODEL_MODE:-server}"
# DOC_REUSE_ANSWER_SERVER=1 (two-server style): the per-round AI docs are built by the ANSWER model,
# so point doc generation at the answer server and DON'T start a separate doc GPU server. Pair with
# DISTRACTOR_MODEL=azure/gpt-5-mini to seed round-0 distractors from a strong api model.
DOC_REUSE_ANSWER_SERVER="${DOC_REUSE_ANSWER_SERVER:-}"
mkdir -p logs

echo "### submitting answer server for '$ANSWER_SERVED' (port $ANSWER_PORT, time $SERVER_TIME) ###"
A_JID=$(sbatch --parsable -t "$SERVER_TIME" -J "ans-$ANSWER_SERVED-bd" \
        --export="ALL,MODEL_NAME=$ANSWER_MODEL_ID,SERVED_MODEL_NAME=$ANSWER_SERVED,EXTRA_VLLM_ARGS=$ANSWER_EXTRA_ARGS,MAX_NUM_SEQS=$ANSWER_MAX_NUM_SEQS,PORT=$ANSWER_PORT" \
        scripts/hotpot-misinfo/server_answer.sh)
D_JID=""
if [[ -n "$DOC_REUSE_ANSWER_SERVER" ]]; then
  echo "  answer=$A_JID  doc=ANSWER SERVER (reused; no separate doc GPU server)"
elif [[ "$DOC_MODEL_MODE" == "server" ]]; then
  D_JID=$(sbatch --parsable -t "$SERVER_TIME" -J "doc-qwen7b-bd" --export="ALL,PORT=$DOCGEN_PORT" \
          scripts/hotpot-misinfo/server_docgen.sh)
  echo "  answer=$A_JID  doc=$D_JID"
else
  echo "  answer=$A_JID  doc=API (DOC_MODEL_MODE=$DOC_MODEL_MODE, no doc GPU server)"
fi

wait_for_server () {
  local jid="$1" log="$2" port="$3" name="$4" url=""
  echo "### waiting for $name (job $jid) to serve on :$port (up to $((WAIT_TRIES*WAIT_SLEEP/60)) min) ###" >&2
  for i in $(seq 1 "$WAIT_TRIES"); do
    local st; st=$(squeue -j "$jid" -h -o %T 2>/dev/null)
    if [[ -z "$st" ]]; then echo "!!! $name job $jid left the queue before serving" >&2; return 1; fi
    if [[ "$st" == "RUNNING" && -f "$log" ]]; then
      url=$(grep -oE "http://[^\" ]+:$port/v1" "$log" 2>/dev/null | head -1)
      if [[ -n "$url" ]] && curl -sf "$url/models" >/dev/null 2>&1; then
        echo "  $name ready: $url" >&2; echo "$url"; return 0
      fi
    fi
    sleep "$WAIT_SLEEP"
  done
  echo "!!! $name (job $jid) never became ready" >&2; return 1
}

A_LOG="logs/slurm-${A_JID}-vllm-answer.out"
A_URL=$(wait_for_server "$A_JID" "$A_LOG" "$ANSWER_PORT" "answer-server")  || { scancel "$A_JID" $D_JID; exit 1; }
D_URL=""
if [[ -n "$DOC_REUSE_ANSWER_SERVER" ]]; then
  D_URL="$A_URL"                       # per-round synthesis runs on the answer model/server
elif [[ -n "$D_JID" ]]; then
  D_LOG="logs/slurm-${D_JID}-vllm-docgen.out"
  D_URL=$(wait_for_server "$D_JID" "$D_LOG" "$DOCGEN_PORT" "docgen-server")  || { scancel "$A_JID" "$D_JID"; exit 1; }
fi

# DOC_MODEL: when reusing the answer server, the doc model IS the answer model.
if [[ -n "$DOC_REUSE_ANSWER_SERVER" ]]; then DOC_MODEL="${DOC_MODEL:-$ANSWER_SERVED}"; else DOC_MODEL="${DOC_MODEL:-qwen2.5-7b-docgen}"; fi

echo "### server(s) up — submitting the distractor-fraction sweep client ###"
CJ=$(sbatch --parsable -t "$CLIENT_TIME" -J "$ANSWER_SERVED-bd-sweep" \
     --export="ALL,VLLM_API_BASE=$A_URL,DOC_VLLM_API_BASE=$D_URL,DOC_MODEL_MODE=$DOC_MODEL_MODE,MODEL=$ANSWER_SERVED,DOC_MODEL=$DOC_MODEL,DISTRACTOR_MODEL=${DISTRACTOR_MODEL:-},DISTRACTOR_MODEL_MODE=${DISTRACTOR_MODEL_MODE:-},DISTRACTOR_VLLM_API_BASE=${DISTRACTOR_VLLM_API_BASE:-},DISTRACTOR_MAX_TOKENS=${DISTRACTOR_MAX_TOKENS:-},LITELLM_API_BASE=${LITELLM_API_BASE:-},VARIANT=${VARIANT:-search},VARIANTS=${VARIANTS:-},MAX_Q=${MAX_Q:-50},NUM_RUNS=${NUM_RUNS:-10},SEED=${SEED:-42},DISTRACTOR_MODE=${DISTRACTOR_MODE:-rewrite},FRACTIONS=${FRACTIONS:-0 0.3 0.5 0.7},TOPICS=${TOPICS:-},DOCS_PER_TOPIC=${DOCS_PER_TOPIC:-},CHARS_PER_DOC=${CHARS_PER_DOC:-500},MAX_TOKENS=${MAX_TOKENS:-512},DOC_MAX_TOKENS=${DOC_MAX_TOKENS:-},NUM_ITERATIONS=${NUM_ITERATIONS:-},DISTRACTOR_PER_RUN=${DISTRACTOR_PER_RUN:-},SHUFFLE_PER_RUN=${SHUFFLE_PER_RUN:-},INITIAL_DOCS=${INITIAL_DOCS:-},NATIVE_FILE=${NATIVE_FILE:-},DISTRACTOR_AVOID_GOLD=${DISTRACTOR_AVOID_GOLD:-},OUTDIR=${OUTDIR:-}" \
     base_hotpotqa_distractors/run_sweep.sh)
echo "  sweep client: $CJ"

REAP=$(sbatch --parsable --dependency=afterany:"$CJ" -p cpu -c 1 --mem=1g -t 00:05:00 \
       -J reap-bd -o logs/reap_%j.out \
       --wrap="scancel $A_JID $D_JID; echo 'reaped servers $A_JID $D_JID after client $CJ'")

cat <<EOF

### launched: base-HotpotQA distractor sweep ('$ANSWER_SERVED') ###
  servers : answer=$A_JID ($A_URL)   docgen=${D_JID:-API ($DOC_MODEL_MODE)} ${D_URL:+($D_URL)}
  client  : $CJ   reaper: $REAP
  sweep   : fractions=${FRACTIONS:-0 0.3 0.5 0.7}  variant=${VARIANT:-search}  mode=${DISTRACTOR_MODE:-rewrite}  doc-backend=$DOC_MODEL_MODE
Monitor:  squeue --me ;  tail -f logs/base_distractor_*.out
Cancel:   scancel $A_JID $D_JID $CJ $REAP
EOF
