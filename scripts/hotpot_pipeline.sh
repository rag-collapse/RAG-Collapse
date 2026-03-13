#!/bin/bash
#SBATCH --job-name=hotpot_pipeline
#SBATCH --output=logs/hotpot_pipeline_%A.out
#SBATCH --error=logs/hotpot_pipeline_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu,superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --constraint=vram40|vram48|vram80
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --account=pi_hzamani_umass_edu

if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
fi

module load conda/latest
conda activate rag
module load cuda/12.6
nvidia-smi

CACHE_DIR="/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache/"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"

MODEL="Qwen/Qwen2.5-14B-Instruct"
OUTDIR="/work/pi_dagarwal_umass_edu/project_4/file_storage/${USER}/experiment_outputs/hotpotqa/$MODEL"
mkdir -p logs "$OUTDIR"

TPARALLEL=1
COMMON="--chars-per-doc 400 --num-runs 10 --tensor-parallel-size $TPARALLEL --split test --max-questions 400"

run() {
  python -u hotpot_pipeline.py --model-mode local --model-name "$MODEL" $COMMON "$@"
}

# Replace All (hybrid: all synth, no corpus docs after round 0)
# run --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 --num-iterations 10 \
#     --output-path "$OUTDIR/hotpot_replace_all.json"

# Replace One
# run --pipeline-variant replace_one --num-iterations 20 \
#     --output-path "$OUTDIR/hotpot_replace_one.json"

# # Search (FAISS + side-buffer)
run --pipeline-variant search --num-iterations 30 \
    --output-path "$OUTDIR/hotpot_search.json"
