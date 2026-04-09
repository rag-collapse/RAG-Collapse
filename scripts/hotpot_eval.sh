#!/bin/bash
# --- SLURM ---
#SBATCH --job-name=hotpot_eval
#SBATCH --output=logs/hotpot_eval_%A.out
#SBATCH --error=logs/hotpot_eval_%A.err
#SBATCH --time=1:00:00
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# Run from submit dir so paths resolve
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
fi

# --- Conda ---
module load conda/latest
conda activate ragenv

# Model subdir, e.g. Qwen/Qwen2.5-14B-Instruct.
MODEL_SUBDIR="${MODEL_SUBDIR:-Qwen/Qwen2.5-14B-Instruct}"
# Input/output on shared file storage.
INDIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/experiment_outputs/hotpotqa/$MODEL_SUBDIR"
OUT_DIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/evaluation_outputs/hotpotqa/$MODEL_SUBDIR"
VISUALS_DIR="visualization_outputs/$MODEL_SUBDIR"
mkdir -p logs "$OUT_DIR" "$VISUALS_DIR"

GT_FILE="/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpot_dev_fullwiki_v1.json"

# possible base settings
# hotpot_search hotpot_replace_one hotpot_replace_all
# hotpot_rerank_lambda0.1

for base in hotpot_search hotpot_replace_one hotpot_replace_all; do
  in_file="$INDIR/${base}.json"
  out_file="$OUT_DIR/${base}_hotpot_eval.json"
  summary_file="$OUT_DIR/${base}_hotpot_eval_summary.json"
  plot_file="$VISUALS_DIR/${base}_avg_f1.png"

  if [[ -f "$in_file" ]]; then
    python -u hotpot_evaluation.py "$in_file" "$out_file" --gt-file "$GT_FILE" --summary-json "$summary_file" --plot-file "$plot_file"
  else
    echo "Skipping missing experiment file: $in_file"
  fi
done
