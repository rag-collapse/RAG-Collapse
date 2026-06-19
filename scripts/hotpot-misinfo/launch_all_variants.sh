#!/bin/bash
# One-command launcher for the misinfo experiment across all 3 variants in parallel,
# sharing ONE pair of vLLM servers. Run this on a LOGIN node (it only submits jobs and
# polls — no heavy compute):
#
#   export GT_FILE=/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpot_dev_fullwiki_v1.json
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

: "${GT_FILE:?set GT_FILE to the native HotpotQA JSON (gold answers)}"
VARIANTS="${VARIANTS:-search replace_one hybrid}"
SERVER_TIME="${SERVER_TIME:-48:00:00}"   # reaper kills servers early; this is just the ceiling
CLIENT_TIME="${CLIENT_TIME:-47:00:00}"   # < server time so servers always outlive clients
ANSWER_PORT="${ANSWER_PORT:-5154}"
DOCGEN_PORT="${DOCGEN_PORT:-5153}"
mkdir -p logs

echo "### submitting shared servers (time limit $SERVER_TIME) ###"
A_JID=$(sbatch --parsable -t "$SERVER_TIME" scripts/hotpot-misinfo/server_answer.sh)
D_JID=$(sbatch --parsable -t "$SERVER_TIME" scripts/hotpot-misinfo/server_docgen.sh)
echo "  answer server job: $A_JID   doc-gen server job: $D_JID"

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

echo "### both servers up — fanning out variants: $VARIANTS ###"
CLIENT_JIDS=()
for v in $VARIANTS; do
  jid=$(sbatch --parsable -t "$CLIENT_TIME" \
        --export=ALL,VLLM_API_BASE="$A_URL",DOC_VLLM_API_BASE="$D_URL",GT_FILE="$GT_FILE",VARIANT="$v",MAX_Q="${MAX_Q:-50}",SEED="${SEED:-42}",TARGET="${TARGET:-final_answer}",MODEL="${MODEL:-qwen2.5-14b}",DOC_MODEL="${DOC_MODEL:-qwen2.5-7b-docgen}",ARMS="${ARMS:-faithful counterfactual freeform}",OUTDIR="${OUTDIR:-hotpot_misinfo_outputs}" \
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

### launched ###
  servers : answer=$A_JID ($A_URL)   docgen=$D_JID ($D_URL)
  clients : ${CLIENT_JIDS[*]}
  reaper  : $REAP
Monitor:  squeue --me
          tail -f logs/hotpot_misinfo_cli_*.out
Cancel everything early:  scancel $A_JID $D_JID ${CLIENT_JIDS[*]} $REAP
EOF
