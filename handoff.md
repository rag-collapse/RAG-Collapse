# Handoff — HotpotQA round-0 distractor experiment (status + open decisions)

The round-0 "distractor document" feature is **built, reviewed, unit-tested, and smoke-tested**. This
doc captures what's done, the two open decisions, and the immediate goal: **study the base HotpotQA
pipeline *with distractor documents* to see how it performs.** Read top to bottom, resolve the two
questions in §2 with the user, then run §3.

---

## 0. TL;DR

- The distractor feature lets you corrupt a fraction of each question's **round-0 *initial retrieved*
  documents** into wrong-answer "distractor" docs (leaving the rest correct), so the **starting**
  answer distribution is wider/partly-wrong instead of a spike on gold. Separate from, and composable
  with, the existing `--inject-round` generated-stream injection.
- **Decision already made:** distractors are **DIVERSE** — each corrupted doc gets its **own distinct**
  invented falsehood (to simulate independently-hallucinated AI docs), via `--distractor-mode rewrite`
  (default), `substitution`, or `native_noise`.
- **Built + validated:** implemented, two independent reviews (find-docs + handoff compliance), 24
  pure-Python tests pass, and a qwen2.5-14b Unity smoke ran end-to-end. **The smoke exposed a quality
  problem with the rewrite distractor (see §4) — that's what the two open questions in §2 are about.**
- **Immediate goal (user):** run the **base HotpotQA pipeline with distractor docs** — i.e. the pure
  distractor arm (faithful synthesis + round-0 distractors), with a matched no-distractor baseline, to
  see how performance shifts.

---

## 1. What's accomplished

**Feature (committed on branch `misinfo-error-compounding`; latest `origin` = `2e2cbd3`):**
- `pipeline/misinfo.py` — `DistractorController` + `DistractorRecord` + helpers `n_to_corrupt`,
  `select_distractor_indices` (prefers gold-bearing docs), `choose_distinct_substitutes`,
  `DISTRACTOR_MODES`. Corrupts docs **in place** so the change persists across rounds in all three
  variants (shared dict objects; search keeps each doc's original embedding so distractors stay
  retrievable). Verified by review.
- `formatters.py` — `get_rewrite_distractor_conversation` (rewrites one passage to imply its own
  invented wrong answer).
- `hotpot_pipeline.py` — flags `--distractor-fraction` (default 0.0=off) and `--distractor-mode
  {rewrite,substitution,native_noise}`; round-0 `prepare_and_apply` (after `doc_llm` is built, before
  the loop); validation (`>0` needs `--gt-file`; `native_noise` needs `--native-hotpot-file`);
  metadata. **`--distractor-fraction 0` is byte-identical to the prior pipeline (guard verified).**
- `hotpot_evaluation.py` — `distractor_aggregates` block (no-op without `initial_distractor` records).
- `tests/test_misinfo.py` — +9 tests (24 total, all pass in `rag-collapse`).
- Per-question schema-additive record `initial_distractor`: a **LIST** `distractors:[{doc_id,
  injected_entity, status}]`, one entry per corrupted doc, plus `{fraction, mode, gold_answer,
  eligible, n_docs, n_corrupted, status}`.

**Diverse-aware metrics** (`distractor_aggregates`, per round incl. round 0): `gold_match_rate`
(accuracy/recovery), `distractor_adoption_rate` (adopts *any* seeded wrong entity), **`distinct_answers`**
(headline diversity signal), `offtarget_rate` (off-gold and off-every-seeded-entity),
`distractor_retrieval_condition`.

**Operational fixes this session (all committed):**
- Server **ports are now configurable** and the launchers actually pass them (was a latent no-op).
  Distinct per model so the experiment coexists with the running baseline and multi-model smokes don't
  collide even on the same node: qwen14b `5164/5163`, mistral7b `5166/5165`, llama8b `5168/5167`,
  deepseek7b `5170/5169` (answer/doc); the DeepSeek **graphite baseline** uses `5154/5153`.
- **`GT_FILE` is now defaulted** to `/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpot_dev_fullwiki_v1.json`
  in `launch_all_variants.sh` + `run_variant_client.sh` — no manual `export` needed (still overridable).
- DeepSeek served **without** `--reasoning-parser` (it corrupts content to byte-level BPE on vLLM 0.20);
  `server_llm.py` `_clean_response()` byte-decodes + strips `<think>` instead. Affects the misinfo
  DeepSeek launcher too.
- Interactive docs: `docs/hotpotqa_experiments.html` (full suite incl. these smoke metrics) and the
  older `docs/distractor_experiment.html`.

**Separately running on Unity (unrelated to this feature):** the **DeepSeek-R1-Distill-Qwen-7B graphite
baseline** (`scripts/deepseek_baseline/launch_deepseek_baseline.sh`) — regenerating the missing
`local_{replace_all,replace_one,search}.json` into `all_experiments/graphite/baseline/...`. It uses 2
GPUs on ports 5154/5153; let it finish or run new work on the per-model ports above.

---

## 2. THE TWO OPEN QUESTIONS (resolve these first)

### Q1 — How to fix the rewrite distractor's effectiveness? (see §4 for the data)
The qwen2.5-14b smoke showed the `rewrite` distractor is only ~⅓ effective per doc and **leaks the
gold answer** into the seeded set (13/75 "distractors" asserted the *correct* answer, which also
inflates `distractor_adoption_rate`). Options:
- **(a) Fix `rewrite` (recommended):** (i) **gold-leak exclusion** — if the discovered entity normalizes
  to the gold, mark the doc `leaky`/failed and drop it from the seeded-entity set + `status=ok` should
  require a non-empty, non-gold entity; (ii) **substitution fallback** — for any corrupted doc where
  rewrite yields no usable wrong answer, fall back to a distinct substituted entity, so every corrupted
  doc becomes a real, diverse distractor while keeping naturalistic rewrite where it works.
- **(b) Switch to pure `substitution`** — reliable distinct wrong entities, exact ground truth, less
  naturalistic text. (Already implemented; `choose_distinct_substitutes` gives diversity.)
- **(c) Proceed as-is** — accept ~1.33 effective distractors/question and the gold-leak (not recommended).

Recommendation: **(a)**, then re-smoke. It preserves the user's "diverse hallucinated docs" intent while
making the corruption actually land.

### Q2 — The arm / baseline matrix for "base HotpotQA with distractors"
`--distractor-fraction` applies to **whichever `ARMS` run**, so it composes with the synthesis arms.
For the user's goal (study the *base* pipeline + distractors) the clean design is:
- **Pure distractor arm:** `ARMS="faithful"` + `--distractor-fraction 0.5` (faithful synthesis = no
  generated-stream injection; the only perturbation is the round-0 distractors).
- **Matched no-distractor baseline:** `ARMS="faithful"` + `--distractor-fraction 0` — *required* to
  interpret the result. (The smoke's `gold_match` looked "flat at 0.20" but there was **no baseline to
  compare against**, so the absolute number is uninterpretable on its own.)
- Optionally sweep `--distractor-fraction` (0 / 0.3 / 0.5 / 0.7) to see the dose-response.
- Keep the injection arms (`counterfactual`/`freeform`, `--distractor-fraction 0`) as a *separate* run
  if/when you want the loop-amplification story — don't conflate them in one run.

Confirm: is the immediate experiment "faithful + distractors vs faithful baseline, sweeping fraction,
across the 4 models and 3 variants"? If so, §3 is the run plan.

---

## 3. Run plan for "base HotpotQA + distractors" (after §2 is resolved)

On Unity (just `git pull origin misinfo-error-compounding` — no env exports needed; `GT_FILE` is
defaulted). Per-model launchers each start their own answer+doc servers on the ports in §1.

**Smoke first (fast — set `NUM_RUNS=2` and a 10-round variant so it's ~3 min, not 21):**
```bash
OUTPUT_BASE=$HOME/misinfo_smoke MAX_Q=20 VARIANTS=hybrid NUM_RUNS=2 ARMS="faithful" \
SERVER_TIME=01:30:00 CLIENT_TIME=01:00:00 DISTRACTOR_FRACTION=0.5 \
  bash scripts/hotpot-misinfo/launch_qwen14b.sh
```
Verify clean answers + that `distractor_aggregates.gold_match` *drops* vs a `DISTRACTOR_FRACTION=0`
run and `distinct_answers` rises. Then smoke the other 3 models (`_mistral7b`/`_llama8b`/`_deepseek7b`).

**Full run** (drop the smoke overrides; per the §2 matrix — pure distractor + matched baseline):
```bash
for L in qwen14b mistral7b llama8b deepseek7b; do
  ARMS="faithful" DISTRACTOR_FRACTION=0.5 bash scripts/hotpot-misinfo/launch_$L.sh   # distractor arm
  ARMS="faithful" DISTRACTOR_FRACTION=0   bash scripts/hotpot-misinfo/launch_$L.sh   # matched baseline
done
```
Outputs: `/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hotpotqa_distractor_experiment/<served>/`.
(Note: same OUTDIR + same filename per variant → the two runs above would overwrite each other; give the
baseline a distinct `OUTPUT_BASE` or filename. Worth wiring a per-run suffix before the full run.)

> ⚠️ GPU reality: the graphite DeepSeek baseline is still running (2 GPUs) and the queue is deep
> (~1500 pending). Smokes use short backfillable servers; the full suite (up to 8 GPUs) will queue.

---

## 4. Smoke results (qwen2.5-14b, the evidence behind Q1)

Run: `search` variant, 20 questions, `NUM_RUNS=10`, `--distractor-fraction 0.5`, `ARMS=faithful`, ports
5164/5163. Completed clean (6000 answers, 0 byte-artifacts, 0 empty). Plumbing works end-to-end.

| Finding | Number | Meaning |
|---|---|---|
| eligible questions | 15/20 | 5 skipped (yes/no answers) |
| corrupted-doc sub-records | 75 | 15 q × 5 docs (0.5 of top-k=10) |
| per-doc status | **53 ok / 22 no_error_produced** | rewrite often yields no discoverable wrong claim |
| empty `injected_entity` | **37/75** | ~half the corruptions carry no usable wrong answer |
| **gold-leak** (`injected==gold`) | **13/75** | failed distractors that *assert the gold* + inflate adoption |
| distinct *real* wrong entities/q | mean **1.33** (0–3); 12/15 ≥1 | only ~⅓ of corruptions become effective distractors |

`distractor_aggregates` (round 0 → 29): `gold_match` 0.200→0.200 (flat — **but no baseline to compare**),
`distractor_adoption` 0.667→0.533 (inflated by gold-leak), `distinct_answers` 1.4→1.2, `offtarget`
0.333→0.467. Conclusion: rewrite needs the Q1(a) fixes before it meaningfully moves the distribution.

Cause: the 7B doc model's rewrite + 7B discovery judge frequently fail or re-extract the gold on hard
multi-hop questions. Model-independent (shared across all answer models) — so don't smoke all 4 models
until Q1 is resolved.

---

## 5. Conceptual background (why this feature exists)

Frame "collapse" as **(a) concentration** (distribution narrows) and **(b) shift** (the mode moves).
- Entity prompts (`umass_data.entity.*`): subjective → wide support → collapse = *concentration*.
- HotpotQA: near-degenerate (one right answer) → only the *shift* off gold is interesting.
- The project's factual runs were flat because round 0 was seeded with **correct-answer docs** — a spike
  on gold; iterating a loop whose fixed point is the truth is a no-op. To see drift you must move the
  start off-gold (**distractors**, this feature) or perturb the loop (**`--inject-round`**).
- Distractors test *static RAG robustness* (round-0 foolability); injection tests *loop amplification*
  (the project's novel thesis). The decisive future experiment: introduce a distractor that fools the
  model at round 0, then **remove it**, and see whether the error persists via the model's own generated
  docs (real compounding) or decays.
- Caveat: hand-forced errors can shade into context-faithfulness rather than emergent compounding; the
  `native_noise`/untargeted control and a matched baseline bound that.

---

## 6. HotpotQA's own distractors (researched)

HotpotQA's official *distractor setting* = 2 gold + 8 bigram-TF-IDF distractor paragraphs, but those
are **answer-absent hard negatives**, not wrong-answer assertions — they widen the start as noise, not as
a competing wrong attractor. Our pipeline retrieves from a **FAISS Wikipedia index** (not the native
`context` field), reading the native JSON only for gold answers + supporting_facts. So wrong-answer
distractors are **created** (rewrite/substitution); the native paragraphs feed the `native_noise` mode.
Native `context` format: list of `[title, [sentences…]]` (web-confirmed: github.com/hotpotqa/hotpot).

---

## 7. Code map (as-built; reference symbols, not line numbers — they shift)

- `hotpot_pipeline.py`: flag parsing + `distractor_enabled` guard; `DistractorController(...).prepare_and_apply(states)`
  at round 0 (after `doc_llm` built, before the loop); metadata + attach. Round loop unchanged.
- `pipeline/misinfo.py`: `DistractorController`, `DistractorRecord`, `n_to_corrupt`,
  `select_distractor_indices`, `choose_distinct_substitutes`, `DISTRACTOR_MODES`, plus the original
  injection machinery (`MisinfoController`, `substitute_in_text`, `choose_substitute`,
  `validate_substitute`, `realized_status`, `load_ground_truth`, `load_native_records`, `normalize`).
- `hotpot_evaluation.py`: `_distractor_metrics_for_iteration` + `distractor_aggregates`.
- `formatters.py`: `get_rewrite_distractor_conversation` + the injection prompts/judges.
- `llm_service/server_llm.py`: `_clean_response` (byte-decode + `<think>` strip) on every answer path.
- Scripts: `scripts/hotpot-misinfo/{launch_all_variants.sh, run_variant_client.sh, launch_<model>.sh,
  server_answer.sh, server_docgen.sh, smoke_test.sh, smoke_all_in_one.sh, RUNBOOK.md}`.
- AI-doc tagging: generated docs `gen_{iter}_{i}` / `url=model_generated`; corpus docs `corpus_{did}`.
  Round-0 distractors are corpus docs mutated in place + tagged `distractor=True`.

---

## 8. Constraints & gotchas

- **Unity GitHub auth:** the Unity clone is HTTPS with no stored credential — the agent **cannot
  pull/push there**. Workflow: commit + push from the laptop clone
  (`C:\Users\riddh\RAG-Collapsement-on-Self-Refined-Generation`), then the **user** pulls on Unity.
- **Envs:** Unity `ragenv` (vllm/faiss/torch). Laptop `rag-collapse` (no GPU/server) is fine for
  `py_compile`, `pytest tests/test_misinfo.py`, wiring checks — use
  `/c/Users/riddh/anaconda3/envs/rag-collapse/python.exe`.
- **2-day SLURM cap** — handled: servers 48h with a reaper, clients 47h (servers outlive clients).
- **vLLM 0.20.0 / SLURM 25.11.** DeepSeek must be served **without** `--reasoning-parser` (detok
  regression) — `_clean_response` handles it; Mistral needs `--tokenizer-mode mistral`. Non-agentic
  variants need no tool-call flags.
- **Canonical params** (comparability): `--num-runs 10`, `--chars-per-doc 500`, answer `--max-num-seqs`
  64 for 14B / 128 for 7-8B, `DEFAULT_ROUNDS` search 30 / replace_one 20 / hybrid 10.
- **Never commit API keys.** `CACHE_DIR` stays hardcoded to
  `/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache` (not `$USER`).
- **Smoke time limits should be short** (e.g. `SERVER_TIME=01:30:00`), not the full 48h — the user
  flagged this.

---

## 9. First steps in a fresh session

1. Read `docs/misinfo_error_compounding.md` (now includes the distractor section) and skim
   `pipeline/misinfo.py` (`DistractorController`).
2. **Resolve §2 Q1 and Q2 with the user.**
3. If Q1=(a): implement gold-leak exclusion + substitution fallback in `pipeline/misinfo.py`, update
   tests, commit on the laptop, have the user pull, then re-smoke qwen14b fast (`NUM_RUNS=2
   VARIANT=hybrid`).
4. Run §3 (pure distractor arm + matched baseline; mind the OUTDIR-overwrite note).
5. Keep `--distractor-fraction 0` byte-identical; the graphite DeepSeek baseline is unrelated — leave it.
