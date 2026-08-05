#!/bin/bash
# Smoke test for the HotpotQA misinformation experiment (M0 parity + M1 injection).
# Runs the CLIENT only (CPU): faiss-cpu reads the index and E5 runs on CPU for a few
# queries, so NO GPU is needed here — only the two vLLM servers need GPUs.
#
#SBATCH -J hotpot-misinfo-smoke
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=96g
#SBATCH -t 02:00:00
#SBATCH -o logs/hotpot_misinfo_smoke_%A.out
#SBATCH -e logs/hotpot_misinfo_smoke_%A.err
#SBATCH --mail-type=END,FAIL
#
# Prereqs — start the two GPU servers first and export their URLs (see RUNBOOK):
#   export VLLM_API_BASE="http://<answer-host>:5154/v1"
#   export DOC_VLLM_API_BASE="http://<docgen-host>:5153/v1"
#   sbatch --export=ALL scripts/hotpot-misinfo/smoke_test.sh

set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR" || exit 1; fi

module load conda/latest
conda activate ragenv

SCRATCH=/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_CACHE="$HF_HOME"
mkdir -p logs

: "${VLLM_API_BASE:?set VLLM_API_BASE (answer server, e.g. http://host:5154/v1)}"
: "${DOC_VLLM_API_BASE:?set DOC_VLLM_API_BASE (doc-gen server, e.g. http://host:5153/v1)}"

CACHE_DIR="${CACHE_DIR:-$SCRATCH/hf_cache}"          # holds queries/{test,train,val}
INDEX_DIR="${INDEX_DIR:-$SCRATCH/hotpotqa_index}"     # ivf.index + docid_map.json
GT_FILE="${GT_FILE:-$SCRATCH/hotpot_dev_fullwiki_v1.json}"
MODEL="${MODEL:-qwen2.5-14b}"
DOC_MODEL="${DOC_MODEL:-qwen2.5-7b-docgen}"
VARIANT="${VARIANT:-search}"
OUT="${OUT:-hotpot_misinfo_outputs/smoke}"
mkdir -p "$OUT"

COMMON=(--vllm-api-base "$VLLM_API_BASE" --model-name "$MODEL"
        --doc-vllm-api-base "$DOC_VLLM_API_BASE" --doc-model-name "$DOC_MODEL"
        --cache-dir "$CACHE_DIR" --index-dir "$INDEX_DIR"
        --pipeline-variant "$VARIANT" --max-questions ${MAX_Q:-5} --num-iterations ${ITERS:-3}
        --num-runs 4 --chars-per-doc 500 --seed 42)

echo "########## arm: faithful (parity) ##########"
python -u hotpot_pipeline.py "${COMMON[@]}" \
  --doc-synthesis-mode faithful --output-path "$OUT/faithful.json"

echo "########## arm: counterfactual ##########"
python -u hotpot_pipeline.py "${COMMON[@]}" \
  --doc-synthesis-mode counterfactual --target-mode final_answer --inject-round 1 \
  --gt-file "$GT_FILE" --output-path "$OUT/counterfactual.json"

echo "########## arm: freeform ##########"
python -u hotpot_pipeline.py "${COMMON[@]}" \
  --doc-synthesis-mode freeform --target-mode final_answer --inject-round 1 \
  --gt-file "$GT_FILE" --output-path "$OUT/freeform.json"

echo "########## arm: distractor (faithful synthesis + round-0 DIVERSE distractors) ##########"
python -u hotpot_pipeline.py "${COMMON[@]}" \
  --doc-synthesis-mode faithful \
  --distractor-fraction "${DISTRACTOR_FRACTION:-0.5}" --distractor-mode "${DISTRACTOR_MODE:-rewrite}" \
  --gt-file "$GT_FILE" --output-path "$OUT/distractor.json"

echo "########## eval (counterfactual) ##########"
python -u hotpot_evaluation.py "$OUT/counterfactual.json" "$OUT/counterfactual_eval.json" \
  --gt-file "$GT_FILE" --summary-json "$OUT/counterfactual_summary.json" \
  --plot-file "$OUT/counterfactual_f1.png"

echo "########## eval (distractor) ##########"
python -u hotpot_evaluation.py "$OUT/distractor.json" "$OUT/distractor_eval.json" \
  --gt-file "$GT_FILE" --summary-json "$OUT/distractor_summary.json" \
  --plot-file "$OUT/distractor_f1.png"

echo "########## checks ##########"
python - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
def load(p):
    with open(p) as f: return json.load(f)

# 1. parity: faithful has NO injection records and no misinfo metadata
faith = load(f"{out}/faithful.json")
assert all("injection" not in q for q in faith["questions"]), "faithful must not carry injection records"
assert "doc_synthesis_mode" not in faith["experiment_metadata"], "faithful metadata must be unchanged"
print("PASS parity: faithful arm is byte-compatible (no injection keys)")

# 2. counterfactual: at least one eligible injection; statuses sane; doc_ids recorded
cf = load(f"{out}/counterfactual.json")
recs = [q["injection"] for q in cf["questions"] if "injection" in q]
assert recs, "counterfactual produced no injection records"
elig = [r for r in recs if r["eligible"]]
print(f"  counterfactual: {len(recs)} records, {len(elig)} eligible; "
      f"statuses={ {r['status'] for r in recs} }")
assert elig, "no eligible counterfactual injections (try more --max-questions)"
ok = [r for r in elig if r["status"] == "ok"]
assert ok, "no eligible injection reached status 'ok'"
assert all(r["injected_entity"] for r in elig), "eligible record missing injected_entity"
assert any(r["injected_doc_ids"] for r in ok), "no injected_doc_ids recorded"
print(f"PASS counterfactual: {len(ok)} questions with status=ok and recorded doc ids")

# 3. freeform: discovery harness ran
ff = load(f"{out}/freeform.json")
ff_elig = [q["injection"] for q in ff["questions"] if q.get("injection", {}).get("eligible")]
assert ff_elig, "no eligible freeform injections"
assert any(r.get("discovered_claims") is not None for r in ff_elig), "freeform discovery produced nothing"
print(f"PASS freeform: {len(ff_elig)} eligible; discovery harness populated claims")

# 4. eval produced injection aggregates
ev = load(f"{out}/counterfactual_eval.json")
agg = ev.get("injection_aggregates")
assert agg and agg["n_eligible"] >= 1, "eval missing injection_aggregates"
print(f"PASS eval: injection_aggregates present "
      f"(n_eligible={agg['n_eligible']}, adoption_rate={agg['adoption_rate']:.3f}); "
      f"asr_by_iteration={agg['asr_by_iteration']}")

# 5. distractor: round-0 distractors recorded + parity (no injection records) + eval aggregates
dd = load(f"{out}/distractor.json")
drecs = [q["initial_distractor"] for q in dd["questions"] if "initial_distractor" in q]
assert drecs, "distractor arm produced no initial_distractor records"
delig = [r for r in drecs if r["eligible"]]
assert delig, "no eligible distractors (try more --max-questions)"
# DIVERSE: at least one question seeded >1 distinct wrong entity across its corrupted docs
diverse = [r for r in delig if len({d.get("injected_entity") for d in r["distractors"] if d.get("injected_entity")}) > 1]
assert all("injection" not in q for q in dd["questions"]), "distractor arm must not carry injection records (faithful synthesis)"
dev = load(f"{out}/distractor_eval.json")
dagg = dev.get("distractor_aggregates")
assert dagg and dagg["n_eligible"] >= 1, "eval missing distractor_aggregates"
print(f"PASS distractor: {len(delig)} eligible; {len(diverse)} with >1 distinct wrong entity; "
      f"gold_match_by_iteration={dagg['gold_match_by_iteration']}")
print("\nALL SMOKE CHECKS PASSED")
PY
