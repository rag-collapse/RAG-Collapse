#!/bin/bash
# Submit the 8 temperature=1.0 / top_p=1.0 co-located distractor sweeps:
#   4 models x { diverse_synth, equal_diverse_synth }, each a self-contained 2-GPU job
#   (launch_colocated.sh) that starts its own answer + qwen2.5-7b-docgen servers and runs the
#   sweep inline — so no GPU ever idles waiting on a separate client (Unity idle-reclaim safe).
#
# 400 questions, 10 runs, 3 rounds, native 2-gold/8-distractor round-0 seeding, gold never
# corrupted (seed 42), gpt-5-mini distractors via the OpenAI API. One sweep per job (each ~<48h).
#
# Run on a LOGIN node from the repo root (needs OPENAI_API_KEY in ./.env):
#   bash base_hotpotqa_distractors/submit_temp1_sweeps.sh            # submit all 8
#   DRYRUN=1 bash base_hotpotqa_distractors/submit_temp1_sweeps.sh   # print sbatch lines only
#   MODES="diverse_synth" bash ...                                   # just the 4 diverse jobs
#   MODELS="qwen14b" bash ...                                        # just one model (both modes)
set -eo pipefail
cd "$(dirname "$0")/.."   # repo root
SCR=/scratch4/workspace/oyilmazel_umass_edu-rag_collapse
mkdir -p logs

# Slurm emails (BEGIN/END/FAIL/TIME_LIMIT come from launch_colocated.sh's #SBATCH --mail-type).
MAIL_USER="${MAIL_USER:-riddhimaan.senapati@graphitehq.com}"

# ---- shared sweep parameters ----
export TEMPERATURE=1.0 TOP_P=1.0
export MAX_Q="${MAX_Q:-400}" NUM_RUNS=10 SEED=42 NUM_ITERATIONS=3
export DISTRACTOR_MODEL=gpt-5-mini DISTRACTOR_MODEL_MODE=api LITELLM_API_BASE=openai
export INITIAL_DOCS=native_distractor NATIVE_FILE="$SCR/hotpot_dev_distractor_v1.json"
export DISTRACTOR_AVOID_GOLD=1
export DOC_MODEL_ID=Qwen/Qwen2.5-7B-Instruct DOC_SERVED=qwen2.5-7b-docgen
export VARIANTS="search hybrid replace_one"
export FRACTIONS="0 0.3 0.5 0.7"      # diverse_synth
export TOPICS="0 1 2 3 4" DOCS_PER_TOPIC=2   # equal_diverse_synth

# ---- per-model config: tag | model_id | served | max_num_seqs | extra_vllm_args | constraint ----
# DeepSeek: NO --reasoning-parser (server_llm.py _clean_response handles the vLLM detok bug); see handoff.md.
MODELS_ALL="qwen14b llama mistral deepseek"
declare -A MID SRV SEQS EXTRA CON
MID[qwen14b]=Qwen/Qwen2.5-14B-Instruct;              SRV[qwen14b]=qwen2.5-14b;                 SEQS[qwen14b]=64;  EXTRA[qwen14b]="";                     CON[qwen14b]="bf16&vram48"
MID[llama]=meta-llama/Llama-3.1-8B-Instruct;         SRV[llama]=llama-3.1-8b;                  SEQS[llama]=128;   EXTRA[llama]="";                      CON[llama]="bf16&vram23"
MID[mistral]=mistralai/Mistral-7B-Instruct-v0.3;     SRV[mistral]=mistral-7b;                  SEQS[mistral]=128; EXTRA[mistral]="--tokenizer-mode mistral"; CON[mistral]="bf16&vram23"
MID[deepseek]=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; SRV[deepseek]=deepseek-r1-distill-qwen-7b; SEQS[deepseek]=128; EXTRA[deepseek]="";                 CON[deepseek]="bf16&vram23"

MODELS="${MODELS:-$MODELS_ALL}"
MODES="${MODES:-diverse_synth equal_diverse_synth}"

# ---- distinct port pair per (model,mode) so co-scheduled jobs on one node never collide ----
port=5200
for m in $MODELS; do
  for mode in $MODES; do
    export ANSWER_MODEL_ID="${MID[$m]}" ANSWER_SERVED="${SRV[$m]}" \
           ANSWER_MAX_NUM_SEQS="${SEQS[$m]}" ANSWER_EXTRA_ARGS="${EXTRA[$m]}" \
           ANSWER_PORT="$port" DOCGEN_PORT="$((port+1))" \
           DISTRACTOR_MODE="$mode" \
           OUTDIR="$HOME/temp1_sweeps/$mode/${SRV[$m]}"
    port=$((port+2))
    jobtag="t1-${SRV[$m]}-${mode%%_*}"     # e.g. t1-qwen2.5-14b-diverse / t1-...-equal
    mkdir -p "$OUTDIR"
    echo ">>> $jobtag  ports ${ANSWER_PORT}/${DOCGEN_PORT}  constraint '${CON[$m]}'  -> $OUTDIR"
    if [[ -n "${DRYRUN:-}" ]]; then
      echo "    DRYRUN: sbatch -J $jobtag --constraint='${CON[$m]}' --mail-user='$MAIL_USER' --export=ALL base_hotpotqa_distractors/launch_colocated.sh"
    else
      sbatch -J "$jobtag" --constraint="${CON[$m]}" --mail-user="$MAIL_USER" --export=ALL \
        base_hotpotqa_distractors/launch_colocated.sh
    fi
  done
done
echo "### done. monitor: squeue --me ;  tail -f logs/colo_*.out ;  ls -R ~/temp1_sweeps ###"
