#!/bin/bash
# Compute compare_sweep summaries for the 1400q baseline-match distractor runs (SKIP_COMPARE was set
# during the sweep, so no summaries exist yet). One summary per (model, mode, variant) — 24 total —
# written next to the arm outputs on /work. High-mem CPU job: compare_sweep loads all 4-5 arm JSONs
# (~600MB each) of a variant simultaneously.
#
#   sbatch base_hotpotqa_distractors/compute_1400q_summaries.sh
#
#SBATCH -J compute-1400q-summaries
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH -t 06:00:00
#SBATCH -o logs/compute_summaries_%j.out
#SBATCH -e logs/compute_summaries_%j.err
#SBATCH --mail-type=END,FAIL
set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR" || exit 1; fi

module load conda/latest
conda activate ragenv

SCR=/scratch4/workspace/oyilmazel_umass_edu-rag_collapse
GT="${GT_FILE:-$SCR/hotpot_dev_fullwiki_v1.json}"
BASE="${BASE_OUTDIR:-/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/baseline_match_temp1}"
MODELS="qwen2.5-14b llama-3.1-8b mistral-7b deepseek-r1-distill-qwen-7b"

n_ok=0; n_fail=0
for mode in diverse_synth equal_diverse_synth; do
  for model in $MODELS; do
    D="$BASE/$mode/$model"
    for var in search replace_one replace_all; do
      files=$(ls "$D"/base_${var}_*.json 2>/dev/null || true)
      if [[ -z "$files" ]]; then echo "skip $mode/$model/$var (no arm files)"; continue; fi
      out="$D/sweep_summary_${var}.json"
      echo "==================== $mode/$model/$var -> $out ===================="
      if python -u base_hotpotqa_distractors/compare_sweep.py --gt-file "$GT" --summary "$out" $files; then
        n_ok=$((n_ok+1))
      else
        echo "!!! FAILED: $mode/$model/$var"; n_fail=$((n_fail+1))
      fi
    done
  done
done
echo "### done: $n_ok summaries written, $n_fail failed. Under $BASE/<mode>/<model>/sweep_summary_<variant>.json ###"
