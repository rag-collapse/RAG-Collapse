#!/bin/bash
# One-command launcher for the misinfo experiment across all 3 variants in parallel,
# sharing ONE pair of vLLM servers. Run this on a LOGIN node (it only submits jobs and
# polls — no heavy compute):
#
#   (GT_FILE defaults to the native HotpotQA JSON; override via env only if it moves)
#   bash scripts/hotpot-misinfo/launch_all_variants.sh
#
# What it does:
#   1. submits the two GPU servers (answer + doc-gen) — 2 GPUs total, shared by all variants
#   2. waits until both are loaded and serving (parses each server log for its URL + curls /models)
#   3. submits one CPU client per variant (search/replace_one/hybrid), each running all 3 arms,
#      all pointed at the same two server URLs (they run fully in parallel)
#   4. submits a reaper job (depends on all clients) that scancels the servers when they finish
#
# Env overrides: VARIANTS, MAX_Q, SEED, TARGET, MODEL, DOC_MODEL, SERVER_TIME, CLIENT_TIME,
#                ARMS, OUTDIR.
set -eo pipefail
cd "$(dirname "$0")/../.." || exit 1     # repo root

# GT_FILE defaults to the native HotpotQA JSON on Unity; override via env if it moves.
GT_FILE="${GT_FILE:-/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpot_dev_fullwiki_v1.json}"
VARIANTS="${VARIANTS:-search replace_one hybrid}"
SERVER_TIME="${SERVER_TIME:-48:00:00}"   # reaper kills servers early; this is just the ceiling
CLIENT_TIME="${CLIENT_TIME:-47:00:00}"   # < server time so servers always outlive clients
ANSWER_PORT="${ANSWER_PORT:-5154}"
DOCGEN_PORT="${DOCGEN_PORT:-5153}"

# Answer model — overridden by the per-model wrappers (launch_<model>.sh). Doc-gen stays
# fixed at Qwen2.5-7B (server_docgen.sh) for all answer models.
ANSWER_MODEL_ID="${ANSWER_MODEL_ID:-Qwen/Qwen2.5-14B-Instruct}"
ANSWER_SERVED="${ANSWER_SERVED:-qwen2.5-14b}"     # served name == client --model-name
ANSWER_EXTRA_ARGS="${ANSWER_EXTRA_ARGS:-}"        # model-specific vLLM flags (e.g. --reasoning-parser deepseek_r1)
ANSWER_MAX_NUM_SEQS="${ANSWER_MAX_NUM_SEQS:-64}"  # 64 for 14B (anti-OOM); 7-8B wrappers set 128
OUTDIR="${OUTDIR:-hotpot_misinfo_outputs}"
mkdir -p logs

echo "### submitting shared servers for answer model '$ANSWER_SERVED' (time limit $SERVER_TIME) ###"
A_JID=$(sbatch --parsable -t "$SERVER_TIME" -J "ans-$ANSWER_SERVED" \
        --export="ALL,MODEL_NAME=$ANSWER_MODEL_ID,SERVED_MODEL_NAME=$ANSWER_SERVED,EXTRA_VLLM_ARGS=$ANSWER_EXTRA_ARGS,MAX_NUM_SEQS=$ANSWER_MAX_NUM_SEQS,PORT=$ANSWER_PORT" \
        scripts/hotpot-misinfo/server_answer.sh)
D_JID=$(sbatch --parsable -t "$SERVER_TIME" -J "doc-qwen7b" --export="ALL,PORT=$DOCGEN_PORT" \
        scripts/hotpot-misinfo/server_docgen.sh)   # no MODEL_NAME exported -> keeps its Qwen2.5-7B default
echo "  answer server job: $A_JID ($ANSWER_MODEL_ID)   doc-gen server job: $D_JID (Qwen2.5-7B)"

# Wait until a server job is RUNNING, its log prints the URL, and /models responds.
# args: <jobid> <logfile> <port> <varname-for-message>  -> prints the URL on stdout
wait_for_server () {
  local jid="$1" log="$2" port="$3" name="$4" url=""
  echo "### waiting for $name (job $jid) to serve on :$port ###" >&2
  for i in $(seq 1 240); do          # up to ~40 min (model load + queue)
    local st; st=$(squeue -j "$jid" -h -o %T 2>/dev/null)
    if [[ -z "$st" ]]; then echo "!!! $name job $jid left the queue before serving" >&2; return 1; fi
    if [[ "$st" == "RUNNING" && -f "$log" ]]; then
      url=$(grep -oE "http://[^\" ]+:$port/v1" "$log" 2>/dev/null | head -1)
      if [[ -n "$url" ]] && curl -sf "$url/models" >/dev/null 2>&1; then
        echo "  $name ready: $url" >&2
        echo "$url"; return 0
      fi
    fi
    sleep 10
  done
  echo "!!! $name (job $jid) never became ready" >&2; return 1
}

A_LOG="logs/slurm-${A_JID}-vllm-answer.out"
D_LOG="logs/slurm-${D_JID}-vllm-docgen.out"
A_URL=$(wait_for_server "$A_JID" "$A_LOG" "$ANSWER_PORT" "answer-server")  || { scancel "$A_JID" "$D_JID"; exit 1; }
D_URL=$(wait_for_server "$D_JID" "$D_LOG" "$DOCGEN_PORT" "docgen-server")  || { scancel "$A_JID" "$D_JID"; exit 1; }

echo "### both servers up — fanning out variants for '$ANSWER_SERVED': $VARIANTS -> $OUTDIR ###"
CLIENT_JIDS=()
for v in $VARIANTS; do
  jid=$(sbatch --parsable -t "$CLIENT_TIME" -J "$ANSWER_SERVED-$v" \
        --export="ALL,VLLM_API_BASE=$A_URL,DOC_VLLM_API_BASE=$D_URL,GT_FILE=$GT_FILE,VARIANT=$v,MAX_Q=${MAX_Q:-50},SEED=${SEED:-42},TARGET=${TARGET:-final_answer},MODEL=$ANSWER_SERVED,DOC_MODEL=${DOC_MODEL:-qwen2.5-7b-docgen},ARMS=${ARMS:-faithful counterfactual freeform},NUM_RUNS=${NUM_RUNS:-10},CHARS_PER_DOC=${CHARS_PER_DOC:-500},MAX_TOKENS=${MAX_TOKENS:-512},DOC_MAX_TOKENS=${DOC_MAX_TOKENS:-512},DISTRACTOR_FRACTION=${DISTRACTOR_FRACTION:-0},DISTRACTOR_MODE=${DISTRACTOR_MODE:-rewrite},OUTDIR=$OUTDIR" \
        scripts/hotpot-misinfo/run_variant_client.sh)
  echo "  variant $v -> client job $jid"
  CLIENT_JIDS+=("$jid")
done

DEP=$(IFS=:; echo "${CLIENT_JIDS[*]}")
REAP=$(sbatch --parsable --dependency=afterany:"$DEP" -p cpu -c 1 --mem=1g -t 00:05:00 \
       -J reap-misinfo-servers -o logs/reap_%j.out \
       --wrap="scancel $A_JID $D_JID; echo 'reaped servers $A_JID $D_JID after clients $DEP'")
echo "  reaper job: $REAP (scancels servers $A_JID $D_JID once all clients finish)"

cat <<EOF

### launched: answer model '$ANSWER_SERVED' ###
  servers : answer=$A_JID ($A_URL)   docgen=$D_JID ($D_URL)
  clients : ${CLIENT_JIDS[*]}
  reaper  : $REAP
  outputs : $OUTDIR
Monitor:  squeue --me
          tail -f logs/hotpot_misinfo_cli_*.out
Cancel everything early:  scancel $A_JID $D_JID ${CLIENT_JIDS[*]} $REAP
EOF
