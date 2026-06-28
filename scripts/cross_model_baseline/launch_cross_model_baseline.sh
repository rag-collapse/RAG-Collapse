#!/bin/bash
# One-command launcher for the CROSS-MODEL baseline on the graphite (umass-entity) dataset.
# Run on a LOGIN node:
#
#   bash scripts/cross_model_baseline/launch_cross_model_baseline.sh
#
# Idea: a MAIN answer model reads documents synthesized from a SIDE answer model's answers (not its
# own). Each round both models answer from the same context; only the SIDE model's answers go through
# the doc-gen model; the resulting docs feed the next round, which both read. MAIN is measured.
#
# Topology — THREE INDEPENDENT, DEDICATED vLLM servers, one per role, each on its own GPU and port:
#   - main-model : Qwen/Qwen2.5-14B-Instruct        -> served qwen2.5-14b                 port 5190
#   - side-model : deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -> served deepseek-r1-distill-qwen-7b port 5191
#                  (NO --reasoning-parser; server_llm.py byte-decodes + strips <think>)
#   - doc-gen    : Qwen/Qwen2.5-7B-Instruct          -> served qwen2.5-7b-doc             port 5192
# Needs 3 GPUs. Ports 5190-5192 avoid the distractor sweeps on 5180-5183.
#
# This run: all 3 variants (replace_all/replace_one/search) at MAX_ITERS=2 rounds each, into
#   <ALL_EXP_BASE>/graphite/cross-model-baseline/<variant>/experiment_outputs/Qwen/Qwen2.5-14B-Instruct/local_<variant>.json
# Env overrides: VARIANTS, MAX_ITERS, MAX_Q, NUM_RUNS, CHARS_PER_DOC, MAX_TOKENS, SIDE_MAX_TOKENS,
#                DOC_MAX_TOKENS, DATASET, SERVER_TIME, CLIENT_TIME, ALL_EXP_BASE, and the model/port knobs.
set -eo pipefail
cd "$(dirname "$0")/../.." || exit 1     # repo root

VARIANTS="${VARIANTS:-replace_all replace_one search}"
SERVER_TIME="${SERVER_TIME:-12:00:00}"
CLIENT_TIME="${CLIENT_TIME:-10:00:00}"
MAIN_PORT="${MAIN_PORT:-5190}"
SIDE_PORT="${SIDE_PORT:-5191}"
DOCGEN_PORT="${DOCGEN_PORT:-5192}"
MAIN_MODEL_ID="${MAIN_MODEL_ID:-Qwen/Qwen2.5-14B-Instruct}"
MAIN_SERVED="${MAIN_SERVED:-qwen2.5-14b}"
SIDE_MODEL_ID="${SIDE_MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
SIDE_SERVED="${SIDE_SERVED:-deepseek-r1-distill-qwen-7b}"
MODEL_SUBDIR="${MODEL_SUBDIR:-Qwen/Qwen2.5-14B-Instruct}"   # output subdir = the MAIN (measured) model
ALL_EXP_BASE="${ALL_EXP_BASE:-/work/pi_dagarwal_umass_edu/project_4/file_storage/all_experiments}"
WAIT_TRIES="${WAIT_TRIES:-1800}"    # 1800 * 10s = 5h; a 3rd dedicated GPU can take a while to schedule
# DOC_ON_MAIN=1 (Option B): generate documents on the MAIN server instead of a separate doc-gen GPU.
# Needs only 2 GPUs (main + side) — avoids the 3rd-GPU starvation on this oversubscribed cluster. The
# doc-writer then becomes the MAIN model (Qwen2.5-14B) rather than the qwen2.5-7b-doc model.
DOC_ON_MAIN="${DOC_ON_MAIN:-}"
mkdir -p logs

if [[ -n "$DOC_ON_MAIN" ]]; then
  echo "### Option B: submitting TWO servers (main + side); doc-gen runs on the MAIN server. time=$SERVER_TIME ###"
else
  echo "### submitting THREE dedicated servers (main + side + doc-gen), time limit $SERVER_TIME ###"
fi
# main answer server (Qwen2.5-14B): 64 max-seqs (anti-OOM on the 14B).
M_JID=$(sbatch --parsable -t "$SERVER_TIME" -J "xm-main-$MAIN_SERVED" \
        --export="ALL,MODEL_NAME=$MAIN_MODEL_ID,SERVED_MODEL_NAME=$MAIN_SERVED,MAX_NUM_SEQS=${MAIN_MAX_NUM_SEQS:-64},PORT=$MAIN_PORT" \
        scripts/hotpot-misinfo/server_answer.sh)
# side answer server (DeepSeek-R1): NO --reasoning-parser (server_llm.py cleans byte-BPE + <think>).
S_JID=$(sbatch --parsable -t "$SERVER_TIME" -J "xm-side-$SIDE_SERVED" \
        --export="ALL,MODEL_NAME=$SIDE_MODEL_ID,SERVED_MODEL_NAME=$SIDE_SERVED,MAX_NUM_SEQS=${SIDE_MAX_NUM_SEQS:-128},PORT=$SIDE_PORT" \
        scripts/hotpot-misinfo/server_answer.sh)
if [[ -n "$DOC_ON_MAIN" ]]; then
  # Option B: no separate doc-gen GPU — documents are generated on the main server.
  D_JID=""
  DOC_SERVED="$MAIN_SERVED"
  echo "  main=$M_JID ($MAIN_MODEL_ID :$MAIN_PORT)  side=$S_JID ($SIDE_MODEL_ID :$SIDE_PORT)  doc=MAIN SERVER (served $MAIN_SERVED, no 3rd GPU)"
else
  # doc-gen server (Qwen2.5-7B as qwen2.5-7b-doc).
  D_JID=$(sbatch --parsable -t "$SERVER_TIME" -J "xm-doc-qwen7b" \
          --export="ALL,SERVED_MODEL_NAME=qwen2.5-7b-doc,PORT=$DOCGEN_PORT" \
          scripts/hotpot-misinfo/server_docgen.sh)
  DOC_SERVED="qwen2.5-7b-doc"
  echo "  main=$M_JID ($MAIN_MODEL_ID :$MAIN_PORT)  side=$S_JID ($SIDE_MODEL_ID :$SIDE_PORT)  doc=$D_JID (qwen2.5-7b-doc :$DOCGEN_PORT)"
fi

# Wait until a server job is RUNNING, its log prints the URL, and /models responds.
wait_for_server () {
  local jid="$1" log="$2" port="$3" name="$4" url=""
  echo "### waiting for $name (job $jid) to serve on :$port ###" >&2
  for i in $(seq 1 "$WAIT_TRIES"); do
    local st; st=$(squeue -j "$jid" -h -o %T 2>/dev/null)
    if [[ -z "$st" ]]; then echo "!!! $name job $jid left the queue before serving" >&2; return 1; fi
    if [[ "$st" == "RUNNING" && -f "$log" ]]; then
      url=$(grep -oE "http://[^\" ]+:$port/v1" "$log" 2>/dev/null | head -1)
      if [[ -n "$url" ]] && curl -sf "$url/models" >/dev/null 2>&1; then
        echo "  $name ready: $url" >&2; echo "$url"; return 0
      fi
    fi
    sleep 10
  done
  echo "!!! $name (job $jid) never became ready" >&2; return 1
}

M_LOG="logs/slurm-${M_JID}-vllm-answer.out"
S_LOG="logs/slurm-${S_JID}-vllm-answer.out"
M_URL=$(wait_for_server "$M_JID" "$M_LOG" "$MAIN_PORT" "main-server")   || { scancel "$M_JID" "$S_JID" ${D_JID:+"$D_JID"}; exit 1; }
S_URL=$(wait_for_server "$S_JID" "$S_LOG" "$SIDE_PORT" "side-server")   || { scancel "$M_JID" "$S_JID" ${D_JID:+"$D_JID"}; exit 1; }
if [[ -n "$D_JID" ]]; then
  D_LOG="logs/slurm-${D_JID}-vllm-docgen.out"
  D_URL=$(wait_for_server "$D_JID" "$D_LOG" "$DOCGEN_PORT" "docgen-server") || { scancel "$M_JID" "$S_JID" "$D_JID"; exit 1; }
else
  D_URL="$M_URL"   # Option B: doc generation runs on the main server
fi

echo "### servers up — fanning out variants: $VARIANTS (rounds=${MAX_ITERS:-2}, doc=$DOC_SERVED) ###"
CLIENT_JIDS=()
for v in $VARIANTS; do
  jid=$(sbatch --parsable -t "$CLIENT_TIME" -J "xmodel-$v" \
        --export="ALL,VLLM_API_BASE=$M_URL,SIDE_VLLM_API_BASE=$S_URL,DOC_VLLM_API_BASE=$D_URL,VARIANT=$v,ALL_EXP_BASE=$ALL_EXP_BASE,MODEL=$MAIN_SERVED,SIDE_MODEL=$SIDE_SERVED,DOC_MODEL=$DOC_SERVED,MODEL_SUBDIR=$MODEL_SUBDIR,MAX_ITERS=${MAX_ITERS:-2},MAX_Q=${MAX_Q:-400},NUM_RUNS=${NUM_RUNS:-10},CHARS_PER_DOC=${CHARS_PER_DOC:-400},MAX_TOKENS=${MAX_TOKENS:-512},SIDE_MAX_TOKENS=${SIDE_MAX_TOKENS:-4096},DOC_MAX_TOKENS=${DOC_MAX_TOKENS:-512},DATASET=${DATASET:-datasets/umass_data.entity.chatgpt.400.jsonl}" \
        scripts/cross_model_baseline/run_cross_model_variant.sh)
  echo "  variant $v -> client job $jid"
  CLIENT_JIDS+=("$jid")
done

DEP=$(IFS=:; echo "${CLIENT_JIDS[*]}")
REAP=$(sbatch --parsable --dependency=afterany:"$DEP" -p cpu -c 1 --mem=1g -t 00:05:00 \
       -J reap-xmodel -o logs/reap_%j.out \
       --wrap="scancel $M_JID $S_JID $D_JID; echo 'reaped servers $M_JID $S_JID $D_JID after clients $DEP'")
echo "  reaper job: $REAP (scancels the servers once all variants finish)"

cat <<EOF

### launched: cross-model baseline (main=$MAIN_SERVED, side=$SIDE_SERVED, doc=$DOC_SERVED) ###
  servers : main=$M_JID ($M_URL)   side=$S_JID ($S_URL)   doc=${D_JID:-MAIN-SERVER} ($D_URL)
  clients : ${CLIENT_JIDS[*]}   (rounds=${MAX_ITERS:-2}, variants: $VARIANTS)
  reaper  : $REAP
  outputs : $ALL_EXP_BASE/graphite/cross-model-baseline/<variant>/experiment_outputs/$MODEL_SUBDIR/local_<variant>.json
Monitor:  squeue --me ;  tail -f logs/xmodel_*.out
Cancel:   scancel $M_JID $S_JID $D_JID ${CLIENT_JIDS[*]} $REAP
EOF
