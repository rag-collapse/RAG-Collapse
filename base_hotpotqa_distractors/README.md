# base_hotpotqa_distractors

A focused, standalone experiment: **how does the base HotpotQA recursive-RAG-collapse loop perform
when a fraction of the round-0 retrieved documents are turned into DIVERSE wrong-answer distractor
documents?** Faithful synthesis only, no generated-stream misinfo injection. The headline question is
whether seeding the *start* off-gold (distractors) moves the answer distribution: does accuracy
(`gold_match`) drop and answer diversity (`distinct_answers`, `offtarget`) rise as the distractor
fraction increases, vs a matched no-distractor baseline.

## Design: reuse, don't fork

This package is a **thin experiment interface over the validated pipeline**, not a re-implementation.
It invokes the existing, smoke-tested `hotpot_pipeline.py` with `--doc-synthesis-mode faithful
--distractor-fraction F` and reuses the validated retrieval/loop/LLM machinery and the
`DistractorController` in `pipeline/misinfo.py`. Re-coding FAISS/E5 retrieval or the round loop would
risk silent divergence from the validated core, so we don't. What this folder *adds* is: the
sweep+baseline runner, a matched-cohort comparison analysis, SLURM scripts on their own ports, and a
test for the rewrite fix below.

| File | Role |
|---|---|
| `launch.sh` | login-node orchestrator: shared answer+doc servers (own ports **5180/5181**) → sweep client → reaper |
| `run_sweep.sh` | CPU client: runs `hotpot_pipeline.py` (faithful) for each fraction incl. **0 = matched baseline**, then `compare_sweep.py` |
| `smoke.sh` | quick smoke (hybrid/10-round, 20 q, num_runs 2, fractions {0,0.5}, short servers, throwaway output) |
| `compare_sweep.py` | matched-cohort comparison: per-round `gold_match` / `distractor_adoption` / `offtarget` / `distinct_answers` for every fraction; writes a summary JSON + prints a dose-response table |
| `test_base_distractors.py` | pure-Python tests for the rewrite Q1 fix (gold-leak exclusion + substitution fallback) |

## How to run (Unity; `GT_FILE` is defaulted, no export needed)

```bash
git pull origin misinfo-error-compounding

# 1. smoke (fast): qwen2.5-14b on ports 5180/5181, won't collide with the running baseline
bash base_hotpotqa_distractors/smoke.sh

# 2. full sweep (default fractions 0 / 0.3 / 0.5 / 0.7, search variant, 50 q)
bash base_hotpotqa_distractors/launch.sh

# other models (own server flags)
ANSWER_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3 ANSWER_SERVED=mistral-7b \
  ANSWER_EXTRA_ARGS="--tokenizer-mode mistral" ANSWER_MAX_NUM_SEQS=128 \
  bash base_hotpotqa_distractors/launch.sh
# DeepSeek: ANSWER_MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B ANSWER_SERVED=deepseek-r1-distill-qwen-7b
#   ANSWER_MAX_NUM_SEQS=128  (NO --reasoning-parser; server_llm.py _clean_response handles the detok bug)
```
Overrides (env): `VARIANT`, `FRACTIONS`, `MAX_Q`, `NUM_RUNS`, `DISTRACTOR_MODE` (`rewrite`|`substitution`|`native_noise`),
`SERVER_TIME`/`CLIENT_TIME`, `ANSWER_PORT`/`DOCGEN_PORT`, `OUTDIR`. Outputs (per fraction, **non-overwriting**)
+ `sweep_summary_<variant>.json` land in `OUTDIR`
(default `/work/.../rsenapati_umass_edu/base_hotpotqa_distractors/<served>/`).

## Metrics (computed over the SAME eligible cohort for every arm)

`compare_sweep.py` scores all fractions, including the fraction-0 baseline, on the **same** set of
distractor-eligible questions (deterministic across arms under one seed), so the comparison is
apples-to-apples:

- **`gold_match`**: fraction of runs answering gold (accuracy / recovery). Expect it to *drop* with fraction.
- **`distractor_adoption`**: fraction adopting *any* seeded wrong entity (gold already excluded; see below).
- **`offtarget`**: off-gold AND off-every-seeded-entity (other drift).
- **`distinct_answers`**: mean distinct normalized answers per question (the headline *diversity* signal). Expect it to *rise*.

## Decisions baked in (from `handoff.md` §2)

**Q1, rewrite effectiveness (fixed in shared `pipeline/misinfo.py::DistractorController._apply_rewrite`):**
the smoke showed the rewrite distractor was only ~⅓ effective per doc and leaked the gold answer
(13/75). Now: **(i) gold-leak exclusion**: a discovered entity that normalizes to the gold is not a
distractor (dropped; `status=ok` requires a non-empty, non-gold entity); **(ii) substitution fallback**: any doc whose rewrite yields no usable wrong answer falls back to a *distinct* substituted entity, so
every corrupted doc becomes a real, diverse distractor while keeping the naturalistic rewrite where it
works. `--distractor-fraction 0` stays byte-identical to the base pipeline. Verified by
`test_base_distractors.py` (gold-leak excluded, all docs get distinct non-gold entities, 2 fall back).

**Q2, arm/baseline matrix:** every run produces BOTH the distractor arm(s) and the matched fraction-0
baseline, with **non-overwriting** filenames (`base_<variant>_f<frac>.json`), and supports a fraction
sweep, fixing the overwrite gotcha and giving the baseline needed to interpret the result.

## Notes
- The `native_noise` mode uses HotpotQA's real non-answer paragraphs (`context` = list of
  `[title, [sentences]]`; web-confirmed). It's a noise control, no asserted wrong entity.
- Reuses the validated servers in `scripts/hotpot-misinfo/`. Honors the repo constraints in
  `handoff.md` §8 (vLLM 0.20 model flags, canonical params, 2-day cap via the reaper, hardcoded CACHE_DIR).

---

## Original HotpotQA paper distractor setting (Yang et al., EMNLP 2018)

The synthetic sweep above corrupts retrieved docs to assert *wrong answers*. The **paper's own
distractor setting** is different: each question ships with **2 gold + 8 TF-IDF distractor**
paragraphs, where the distractors are *answer-absent hard negatives* (related but don't state the
answer). This second experiment runs **that** setting through the recursive loop, seeding round 0
from the native context, no synthesis of distractors.

**How it works.** `hotpot_pipeline.py --initial-docs native_distractor --native-hotpot-file <distractor.json>`
seeds round 0 from each question's native `context` (each paragraph a doc, tagged gold/distractor),
bypassing FAISS. `--distractor-gold-only` keeps just the 2 gold paragraphs (the contrast).
`--initial-docs faiss` (default) is byte-identical to the original pipeline. Native seeding and the
synthetic `--distractor-fraction` are mutually exclusive.

**Data.** Needs `hotpot_dev_distractor_v1.json` (the *distractor* setting, gold guaranteed present),
NOT the fullwiki file. The official host `curtis.ml.cmu.edu` is **dead**, so the identical dev data is
reconstructed from HuggingFace (`hotpot_qa`, config `distractor`, split `validation` = the 7,405 dev
questions) by `fetch_distractor_file.py`. `launch_native.sh` runs this automatically on the login node if
the file is missing (needs internet + the `ragenv` env); or generate it manually:
`python base_hotpotqa_distractors/fetch_distractor_file.py <out_path>`.

**Recommended variants.** `replace_one` (default, starts with all 10 native docs, replaces one slot
per round) and `hybrid` carry the native context forward. **`search` caveat:** with native seeding the
search universe is the question's own 10 paragraphs (embedded), growing with generated docs, it does
*not* re-retrieve from Wikipedia. So native seeding is most meaningful in `replace_one`/`hybrid`.

**Run (Unity, login node):**
```bash
bash base_hotpotqa_distractors/smoke_native.sh        # fast: hybrid, 20 q, num_runs 2, both arms
bash base_hotpotqa_distractors/launch_native.sh       # full: replace_one, distractor + gold-only
```
Each run emits both arms `native_<variant>_{distractor,goldonly}.json` (+ per-round F1/EM via
`hotpot_evaluation.py`) and `compare_native.py` prints the per-round `gold_match`/`distinct_answers`
for distractor-setting vs gold-only over the shared cohort. Dedicated ports **5186/5187**.

**Eval note.** Native distractors are answer-absent (no seeded wrong entity), so there is no
`distractor_adoption` here. The read is accuracy degradation over rounds and distractor-setting vs
gold-only. `compare_native.py` + per-arm `hotpot_evaluation.py` (token F1/EM) cover it.

Files: `launch_native.sh`, `run_native.sh`, `smoke_native.sh`, `compare_native.py`, `test_native_seed.py`.
