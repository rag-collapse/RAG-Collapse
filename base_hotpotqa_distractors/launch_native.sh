#!/bin/bash
# Launcher for the ORIGINAL HotpotQA distractor-setting experiment in the recursive loop.
# Run on a LOGIN node:
#   bash base_hotpotqa_distractors/launch_native.sh                       # qwen2.5-14b
#   ANSWER_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3 ANSWER_SERVED=mistral-7b \
#     ANSWER_EXTRA_ARGS="--tokenizer-mode mistral" ANSWER_MAX_NUM_SEQS=128 bash ...   # other models
#
# Starts a shared answer + Qwen2.5-7B doc server on dedicated ports 5186/5187 (coexists with the
# graphite baseline 5154/5153, misinfo 5164-5170, synthetic-distractor 5180/5181), ensures the
# paper's distractor-setting file is present (downloads on this login node), submits the
# native-distractor client (distractor setting + gold-only contrast), and a reaper. Reuses the
# validated servers in scripts/hotpot-misinfo/.
set -eo pipefail
cd "$(dirname "$0")/.." || exit 1          # repo root
SCR=/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu

SERVER_TIME="${SERVER_TIME:-24:00:00}"
CLIENT_TIME="${CLIENT_TIME:-23:00:00}"
ANSWER_PORT="${ANSWER_PORT:-5186}"
DOCGEN_PORT="${DOCGEN_PORT:-5187}"
ANSWER_MODEL_ID="${ANSWER_MODEL_ID:-Qwen/Qwen2.5-14B-Instruct}"
ANSWER_SERVED="${ANSWER_SERVED:-qwen2.5-14b}"
ANSWER_EXTRA_ARGS="${ANSWER_EXTRA_ARGS:-}"     # e.g. "--tokenizer-mode mistral"; DeepSeek: leave empty
ANSWER_MAX_NUM_SEQS="${ANSWER_MAX_NUM_SEQS:-64}"  # 64 for 14B; 128 for 7-8B
NATIVE_FILE="${NATIVE_FILE:-$SCR/hotpot_dev_distractor_v1.json}"
mkdir -p logs

# Ensure the paper's distractor-setting file is present. The official host
# (curtis.ml.cmu.edu) is dead, so we reconstruct the IDENTICAL dev data from HuggingFace
# (hotpot_qa, distractor/validation). Generate here — login nodes have internet + the env;
# compute nodes usually have neither.
if [[ ! -f "$NATIVE_FILE" ]]; then
  echo "Generating $NATIVE_FILE from HuggingFace hotpot_qa (distractor/validation) ..."
  module load conda/latest >/dev/null 2>&1 || true
  conda activate ragenv >/dev/null 2>&1 || true
  HF_HOME="$SCR/hf_cache" HF_HUB_CACHE="$SCR/hf_cache" \
    python "$(dirname "$0")/fetch_distractor_file.py" "$NATIVE_FILE" \
    || { echo "ERROR: failed to generate $NATIVE_FILE (need internet + the ragenv env on this login node)" >&2; exit 1; }
fi

echo "### submitting shared servers for '$ANSWER_SERVED' (ports $ANSWER_PORT/$DOCGEN_PORT, time $SERVER_TIME) ###"
A_JID=$(sbatch --parsable -t "$SERVER_TIME" -J "ans-$ANSWER_SERVED-nd" \
        --export="ALL,MODEL_NAME=$ANSWER_MODEL_ID,SERVED_MODEL_NAME=$ANSWER_SERVED,EXTRA_VLLM_ARGS=$ANSWER_EXTRA_ARGS,MAX_NUM_SEQS=$ANSWER_MAX_NUM_SEQS,PORT=$ANSWER_PORT" \
        scripts/hotpot-misinfo/server_answer.sh)
D_JID=$(sbatch --parsable -t "$SERVER_TIME" -J "doc-qwen7b-nd" --export="ALL,PORT=$DOCGEN_PORT" \
        scripts/hotpot-misinfo/server_docgen.sh)
echo "  answer=$A_JID  doc=$D_JID"

wait_for_server () {
  local jid="$1" log="$2" port="$3" name="$4" url=""
  echo "### waiting for $name (job $jid) to serve on :$port ###" >&2
  for i in $(seq 1 240); do
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

A_LOG="logs/slurm-${A_JID}-vllm-answer.out"
D_LOG="logs/slurm-${D_JID}-vllm-docgen.out"
A_URL=$(wait_for_server "$A_JID" "$A_LOG" "$ANSWER_PORT" "answer-server")  || { scancel "$A_JID" "$D_JID"; exit 1; }
D_URL=$(wait_for_server "$D_JID" "$D_LOG" "$DOCGEN_PORT" "docgen-server")  || { scancel "$A_JID" "$D_JID"; exit 1; }

echo "### both servers up — submitting native-distractor client ###"
CJ=$(sbatch --parsable -t "$CLIENT_TIME" -J "$ANSWER_SERVED-nd" \
     --export="ALL,VLLM_API_BASE=$A_URL,DOC_VLLM_API_BASE=$D_URL,MODEL=$ANSWER_SERVED,DOC_MODEL=${DOC_MODEL:-qwen2.5-7b-docgen},VARIANT=${VARIANT:-replace_one},MAX_Q=${MAX_Q:-50},NUM_RUNS=${NUM_RUNS:-10},SEED=${SEED:-42},NATIVE_FILE=$NATIVE_FILE,ARMS=${ARMS:-distractor goldonly},CHARS_PER_DOC=${CHARS_PER_DOC:-500},MAX_TOKENS=${MAX_TOKENS:-512},DOC_MAX_TOKENS=${DOC_MAX_TOKENS:-512},OUTDIR=${OUTDIR:-}" \
     base_hotpotqa_distractors/run_native.sh)
echo "  client: $CJ"

REAP=$(sbatch --parsable --dependency=afterany:"$CJ" -p cpu -c 1 --mem=1g -t 00:05:00 \
       -J reap-nd -o logs/reap_%j.out \
       --wrap="scancel $A_JID $D_JID; echo 'reaped servers $A_JID $D_JID after client $CJ'")

cat <<EOF

### launched: original-HotpotQA distractor setting ('$ANSWER_SERVED') ###
  servers : answer=$A_JID ($A_URL)   docgen=$D_JID ($D_URL)
  client  : $CJ   reaper: $REAP
  arms    : ${ARMS:-distractor goldonly}   variant=${VARIANT:-replace_one}
Monitor:  squeue --me ;  tail -f logs/base_native_*.out
Cancel:   scancel $A_JID $D_JID $CJ $REAP
EOF
