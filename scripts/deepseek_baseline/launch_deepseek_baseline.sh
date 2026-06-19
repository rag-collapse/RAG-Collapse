#!/bin/bash
# One-command launcher for the DeepSeek-R1-Distill-Qwen-7B graphite BASELINE
# (the missing local_replace_all / local_replace_one / local_search raw outputs).
# Run on a LOGIN node:
#
#   bash scripts/deepseek_baseline/launch_deepseek_baseline.sh
#
# What it does (same proven topology as the misinfo launcher — 2 shared GPUs):
#   1. starts the DeepSeek answer server (reasoning-parser deepseek_r1) and a Qwen2.5-7B
#      doc server (served name "qwen2.5-7b-doc", as in the canonical baseline metadata),
#      reusing the generic servers in scripts/hotpot-misinfo/;
#   2. waits until both are serving (parses each server log for its URL + curls /models);
#   3. fans out one CPU client per variant (replace_all / replace_one / search), all
#      sharing the two servers, reproducing the baseline config (num-runs 10,
#      chars-per-doc 400, 400 questions, top_k 10 / chunk 500 / overlap 50);
#   4. submits a reaper (depends on all clients) that scancels the servers when they finish.
#
# Outputs: <OUTDIR>/local_<variant>.json  (default OUTDIR below).
# Env overrides: VARIANTS, OUTDIR, MAX_Q, NUM_RUNS, CHARS_PER_DOC, MAX_TOKENS, DOC_MAX_TOKENS,
#                DATASET, SERVER_TIME, CLIENT_TIME.
set -eo pipefail
cd "$(dirname "$0")/../.." || exit 1     # repo root

VARIANTS="${VARIANTS:-replace_all replace_one search}"
SERVER_TIME="${SERVER_TIME:-48:00:00}"
CLIENT_TIME="${CLIENT_TIME:-47:00:00}"
ANSWER_PORT="${ANSWER_PORT:-5154}"
DOCGEN_PORT="${DOCGEN_PORT:-5153}"
OUTDIR="${OUTDIR:-/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/experiment_outputs/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
mkdir -p logs

echo "### submitting shared servers (DeepSeek answer + Qwen2.5-7B doc), time limit $SERVER_TIME ###"
A_JID=$(sbatch --parsable -t "$SERVER_TIME" -J "ans-deepseek-r1-7b" \
        --export="ALL,MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,SERVED_MODEL_NAME=deepseek-r1-distill-qwen-7b,EXTRA_VLLM_ARGS=--reasoning-parser deepseek_r1,MAX_NUM_SEQS=128" \
        scripts/hotpot-misinfo/server_answer.sh)
D_JID=$(sbatch --parsable -t "$SERVER_TIME" -J "doc-qwen7b" \
        --export="ALL,SERVED_MODEL_NAME=qwen2.5-7b-doc" \
        scripts/hotpot-misinfo/server_docgen.sh)
echo "  answer server job: $A_JID (DeepSeek-R1-Distill-Qwen-7B)   doc server job: $D_JID (Qwen2.5-7B as qwen2.5-7b-doc)"

# Wait until a server job is RUNNING, its log prints the URL, and /models responds.
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

echo "### both servers up — fanning out baseline variants: $VARIANTS -> $OUTDIR ###"
CLIENT_JIDS=()
for v in $VARIANTS; do
  jid=$(sbatch --parsable -t "$CLIENT_TIME" -J "ds-baseline-$v" \
        --export="ALL,VLLM_API_BASE=$A_URL,DOC_VLLM_API_BASE=$D_URL,VARIANT=$v,OUTDIR=$OUTDIR,MODEL=deepseek-r1-distill-qwen-7b,DOC_MODEL=qwen2.5-7b-doc,MAX_Q=${MAX_Q:-400},NUM_RUNS=${NUM_RUNS:-10},CHARS_PER_DOC=${CHARS_PER_DOC:-400},MAX_TOKENS=${MAX_TOKENS:-4096},DOC_MAX_TOKENS=${DOC_MAX_TOKENS:-512},DATASET=${DATASET:-datasets/umass_data.entity.chatgpt.400.jsonl}" \
        scripts/deepseek_baseline/run_baseline_variant.sh)
  echo "  variant $v -> client job $jid"
  CLIENT_JIDS+=("$jid")
done

DEP=$(IFS=:; echo "${CLIENT_JIDS[*]}")
REAP=$(sbatch --parsable --dependency=afterany:"$DEP" -p cpu -c 1 --mem=1g -t 00:05:00 \
       -J reap-ds-baseline -o logs/reap_%j.out \
       --wrap="scancel $A_JID $D_JID; echo 'reaped servers $A_JID $D_JID after clients $DEP'")
echo "  reaper job: $REAP (scancels servers once all variants finish)"

cat <<EOF

### launched: DeepSeek-R1-Distill-Qwen-7B graphite baseline ###
  servers : answer=$A_JID ($A_URL)   docgen=$D_JID ($D_URL)
  clients : ${CLIENT_JIDS[*]}
  reaper  : $REAP
  outputs : $OUTDIR/local_{replace_all,replace_one,search}.json
Monitor:  squeue --me
          tail -f logs/ds_baseline_*.out
Cancel everything early:  scancel $A_JID $D_JID ${CLIENT_JIDS[*]} $REAP
EOF
