#!/bin/bash
# Submit temperature=1.0/top_p=1.0 distractor sweeps that MATCH oz03's baseline HotpotQA config:
#   1400 questions, FULL per-variant rounds (search=30 / replace_one=20 / replace_all=10), 10 runs,
#   native 2-gold/8-distractor round-0 seeding, gold never corrupted (seed 42), gpt-5-mini distractors.
#   "replace_all" is the hybrid mechanism under the baseline's name.
#
# ONE JOB PER ARM = (model x mode x variant x fraction/topic). A full-round sweep of all fractions in
# one variant would blow past the 48h wall (search ~20h PER fraction at 1400q), so each arm is its own
# co-located 2-GPU job (~= one baseline run, comfortably <48h). SKIP_COMPARE=1 skips the in-job
# comparison; run compare_sweep.py post-hoc over each OUTDIR once all arms land.
#
# Default = Qwen2.5-14B pilot (both modes, all variants/fractions). Run on a LOGIN node (needs OPENAI_API_KEY in .env):
#   bash base_hotpotqa_distractors/submit_baseline_match_sweeps.sh                 # qwen14b pilot (27 jobs)
#   DRYRUN=1 bash ...                                                              # print sbatch lines only
#   MODELS="qwen14b llama mistral deepseek" bash ...                              # full matrix (~108 jobs)
#   MODES="diverse_synth" VARIANTS_LIST="search" bash ...                         # a slice
set -eo pipefail
cd "$(dirname "$0")/.."   # repo root
SCR=/scratch4/workspace/oyilmazel_umass_edu-rag_collapse
mkdir -p logs
MAIL_USER="${MAIL_USER:-riddhimaan.senapati@graphitehq.com}"

# ---- shared params (match the baseline) ----
export TEMPERATURE=1.0 TOP_P=1.0
export MAX_Q="${MAX_Q:-1400}" NUM_RUNS=10 SEED=42
unset NUM_ITERATIONS    # IMPORTANT: unset -> hotpot_pipeline uses DEFAULT_ROUNDS per variant (30/20/10)
export DISTRACTOR_MODEL=gpt-5-mini DISTRACTOR_MODEL_MODE=api LITELLM_API_BASE=openai
export INITIAL_DOCS=native_distractor NATIVE_FILE="$SCR/hotpot_dev_distractor_v1.json"
export DISTRACTOR_AVOID_GOLD=1
export DOC_MODEL_ID=Qwen/Qwen2.5-7B-Instruct DOC_SERVED=qwen2.5-7b-docgen
export DOCS_PER_TOPIC=2
export SKIP_COMPARE=1   # per-arm jobs; compare_sweep.py is run post-hoc over each OUTDIR

# ---- per-model config ----
declare -A MID SRV SEQS EXTRA CON
MID[qwen14b]=Qwen/Qwen2.5-14B-Instruct;              SRV[qwen14b]=qwen2.5-14b;                 SEQS[qwen14b]=64;  EXTRA[qwen14b]="";                     CON[qwen14b]="bf16&vram48"
MID[llama]=meta-llama/Llama-3.1-8B-Instruct;         SRV[llama]=llama-3.1-8b;                  SEQS[llama]=128;   EXTRA[llama]="";                      CON[llama]="bf16&vram23"
MID[mistral]=mistralai/Mistral-7B-Instruct-v0.3;     SRV[mistral]=mistral-7b;                  SEQS[mistral]=128; EXTRA[mistral]="--tokenizer-mode mistral"; CON[mistral]="bf16&vram23"
MID[deepseek]=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; SRV[deepseek]=deepseek-r1-distill-qwen-7b; SEQS[deepseek]=128; EXTRA[deepseek]="";                 CON[deepseek]="bf16&vram23"

MODELS="${MODELS:-qwen14b}"
MODES="${MODES:-diverse_synth equal_diverse_synth}"
VARIANTS_LIST="${VARIANTS_LIST:-search replace_one replace_all}"

port="${BASE_PORT:-5300}"
n=0
for m in $MODELS; do
  for mode in $MODES; do
    if [[ "$mode" == "equal_diverse_synth" ]]; then arms="0 1 2 3 4"; kind=t; else arms="0 0.3 0.5 0.7"; kind=f; fi
    for var in $VARIANTS_LIST; do
      # per-variant walltime headroom (search 30r is the long pole; ~20h/arm at 1400q)
      case "$var" in search) WT=48:00:00;; replace_one) WT=36:00:00;; *) WT=24:00:00;; esac
      for a in $arms; do
        export ANSWER_MODEL_ID="${MID[$m]}" ANSWER_SERVED="${SRV[$m]}" \
               ANSWER_MAX_NUM_SEQS="${SEQS[$m]}" ANSWER_EXTRA_ARGS="${EXTRA[$m]}" \
               ANSWER_PORT="$port" DOCGEN_PORT="$((port+1))" \
               DISTRACTOR_MODE="$mode" VARIANTS="$var" \
               OUTDIR="$HOME/baseline_match_temp1/$mode/${SRV[$m]}"
        if [[ "$kind" == t ]]; then export TOPICS="$a"; unset FRACTIONS; else export FRACTIONS="$a"; unset TOPICS; fi
        port=$((port+2))
        mkdir -p "$OUTDIR"
        tag="bm-${SRV[$m]}-${mode%%_*}-${var}-${kind}${a}"
        echo ">>> $tag  ports ${ANSWER_PORT}/${DOCGEN_PORT}  con '${CON[$m]}'  t=$WT  -> $OUTDIR/base_${var}_${kind}${a}.json"
        if [[ -n "${DRYRUN:-}" ]]; then
          echo "    DRYRUN: sbatch -J $tag --constraint='${CON[$m]}' -t $WT --mail-user=$MAIL_USER --export=ALL base_hotpotqa_distractors/launch_colocated.sh"
        else
          sbatch -J "$tag" --constraint="${CON[$m]}" -t "$WT" --mail-user="$MAIL_USER" --export=ALL \
            base_hotpotqa_distractors/launch_colocated.sh
        fi
        n=$((n+1))
      done
    done
  done
done
echo "### submitted $n arm-job(s). monitor: squeue --me ; ls -R ~/baseline_match_temp1 ###"
echo "### post-hoc per (mode,model): python base_hotpotqa_distractors/compare_sweep.py --gt-file <gt> --summary <out> <OUTDIR>/base_<variant>_*.json ###"
