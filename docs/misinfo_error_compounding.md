# Misinformation Document Synthesis — Error-Compounding Experiment

> **Interactive walkthroughs (self-contained HTML; open in a browser):**
> - [`docs/hotpotqa_smoke_results.html`](hotpotqa_smoke_results.html) — **beginner-friendly consolidated
>   results dashboard**: a plain-language primer + glossary, and every smoke/validation run this session
>   (misinfo, synthetic distractor + fix, DeepSeek baseline, native distractor setting across 4 models).
> - [`docs/hotpotqa_experiments.html`](hotpotqa_experiments.html) — the **full suite** concept explainer:
>   base recursive-RAG loop, error-compounding (3 synthesis modes + 3 targets + injection timing), the
>   round-0 diverse distractor experiment, smoke metrics, and sources.
> - [`docs/original_vs_distractor.html`](original_vs_distractor.html) — original HotpotQA vs the
>   distractor experiment, side by side.
> - [`docs/distractor_experiment.html`](distractor_experiment.html) — the original page focused on the
>   3 synthesis modes, 3 targets, injection timing, and metrics.

Tests whether **factual errors compound across recursive RAG rounds** on HotpotQA — the
inference-time, retrieval-mediated analog of hallucination snowballing. The existing pipeline
synthesizes a *faithful* condensation of the correct answer each round, which is a confound
(later rounds get the answer pre-assembled, making questions easier and masking degradation,
as noted in the paper's Limitations). This experiment lets the synthesizer inject **plausible
misinformation** instead, behind default-off flags, and tracks whether a seeded error
propagates, amplifies, or is corrected once it is laundered into a synthetic document and
re-retrieved.

> Scope: this is a distinct construct from the project's diversity "collapse" — it measures
> **convergence toward a *wrong* attractor**, not loss of variety. Keep it separate from the
> diversity-collapse metric story.

## Synthesis modes (`--doc-synthesis-mode`)

| Mode | Behavior | Ground truth |
|---|---|---|
| `faithful` *(default)* | Unchanged baseline; this machinery is never invoked. **Byte-identical control arm.** | — |
| `counterfactual` | Substitute the gold (or bridge) entity in the answer with a same-type wrong entity *before* synthesis. Treatment vs. control then differ only in the entity. | exact (the injected entity) |
| `freeform` | Instruct the synthesizer to invent one plausible false claim; a judge call then *discovers* what it injected. | LLM-judged claim |

## Distractor targets (`--target-mode`)

| Target | What is corrupted | Notes |
|---|---|---|
| `final_answer` *(default)* | the gold answer entity | needs only the gold answer (`--gt-file`) |
| `intermediate_hop` | a bridge entity in a multi-hop chain | needs `--native-hotpot-file` (supporting_facts); derives the *implied* wrong answer; bridge-only cohort — manually spot-check before trusting `implied_match_rate` |
| `untargeted` | one answer-irrelevant false detail | control: separates "any falsehood destabilizes" from "answer-relevant falsehood" |

## CLI flags (all default-off; faithful == current pipeline)

```
--doc-synthesis-mode {faithful,counterfactual,freeform}   default: faithful
--target-mode {final_answer,intermediate_hop,untargeted}  default: final_answer
--inject-round INT          default: 1   # the it whose synthesized doc is corrupted
--inject-every-round              # stress arm: corrupt every round's doc
--seed INT                        # global RNG; substitute selection uses a separate per-question RNG
--gt-file PATH                    # required when mode != faithful
--native-hotpot-file PATH         # required for target-mode intermediate_hop
```

**Injection timing:** the document synthesized at the end of round `it` first enters the
context at round `it+1`. With `--inject-round 1`, round 1's answer→doc is corrupted, so the
seeded error is present in the context from round 2 onward. (Round 0 is always clean original
docs — there is no AI doc yet to corrupt; the first AI doc, from round 0's answer, is faithful
unless `--inject-round 0`.)

**Reproducibility:** with `--seed`, global RNGs are seeded once; substitute selection uses a
separate per-question RNG keyed on `(seed, query_id)` so it never perturbs the global sequence
that drives answer sampling — guaranteeing the control and treatment arms select **identical
answers** under the same seed. The only varied factor is factual correctness.

## Ground-truth logging

Each question gets a schema-additive `injection` record in the experiment JSON (absent in
faithful runs):

```json
"injection": {
  "mode": "counterfactual", "target": "final_answer", "inject_round": 1, "seed": 42,
  "gold_answer": "Newport", "eligible": true, "status": "ok",
  "injected_entity": "Claremont", "substitute_candidates": ["..."],
  "injected_doc_ids": ["gen_2_0"],
  "discovered_claims": null,                 // freeform only
  "bridge_entity": null, "implied_answer": null   // hop only
}
```
`status` ∈ `ok | leaky_gold | not_realized | gold_absent_in_answer | no_valid_substitute |
no_error_produced | skipped_*` (e.g. yes/no answers and non-bridge questions are skipped).

## Metrics (`hotpot_evaluation.py`, injection-aware; no-op without injection records)

Per round, for eligible injected questions (matching uses the same normalizer as injection):

| Metric | Meaning |
|---|---|
| `retrieval_condition` | injected doc id present in this round's retrieved context (PoisonedRAG retrieval condition) |
| `injected_match_rate` (**ASR**) | fraction of runs whose answer adopts the injected entity (generation condition) |
| `gold_match_rate` | fraction matching gold (recovery signal) |
| `implied_match_rate` | hop only: fraction matching the derived implied-wrong answer (compounding) |
| `doc_propagation` | injected entity reappears in a newly synthesized doc (self-reinforcement) |
| `amplification_count` | distinct off-target answers (string-level divergence proxy) |

Aggregates: per-round ASR / gold / retrieval / propagation curves, **adoption rate**,
**mean time-to-adoption**, **mean persistence**, **recovery rate**, and an ASR-vs-gold plot.

## Round-0 initial-document distractors (`--distractor-fraction`)

A separate, composable knob that corrupts a fraction of each question's **round-0 *initial
retrieved* documents** into wrong-answer "distractor" docs (leaving the rest correct), so the
**starting** answer distribution is wider and partly-wrong instead of a narrow spike on gold.
This is independent of `--inject-round` (which corrupts the *generated* docs later); the two can
be combined. Motivation: the faithful pipeline seeds round 0 with correct-answer docs, so the
loop starts on the gold attractor and can't drift — distractors move the start off-gold.

**DIVERSE by design:** each corrupted doc gets its **own distinct falsehood** (not one shared
wrong entity), to simulate the spread of independently-hallucinated AI documents and the answer
diversity that produces.

| Flag | Meaning |
|---|---|
| `--distractor-fraction FLOAT` | fraction of round-0 docs to corrupt (0 = off, default). Requires `--gt-file`. |
| `--distractor-mode {rewrite,substitution,native_noise,diverse_synth}` | `rewrite` *(default)*: doc-LLM rewrites each passage to imply its own invented wrong answer (independent per-doc, so wrong answers tend to converge → low round-0 diversity). `substitution`: a distinct same-type wrong entity per doc (exact ground truth; reuses the substitute judge). `native_noise`: real non-answer HotpotQA paragraphs (control, no asserted wrong answer; needs `--native-hotpot-file`). `diverse_synth`: **coordinated** — one call proposes K *mutually-distinct* wrong answers, then synthesizes a full naturalistic document per distinct answer (reuses the faithful create-document prompt). Maximizes round-0 answer diversity; **keeps ≥1 gold doc** intact so collapse toward/away from gold is visible. Pair with a strong doc model via `--doc-model-mode api --doc-model-name openai/<model>`. |
| `--doc-model-mode {server,api,local}` | Backend for document/distractor generation (default `server` = vLLM, needs `--doc-vllm-api-base`). `api` routes doc generation through the keymaker LiteLLM proxy (`API_KEY` from env or repo-root `.env`) — use a strong model id for `diverse_synth`, e.g. `azure/gpt-5-mini` (verified) or `openai/claude-sonnet-4-6`. The answerer model is unaffected. |

> **Keymaker model caveats (verified with `test_keymaker_litellm.py` + `ProprietaryLLM`).**
> Both `azure/gpt-5-mini` and `openai/gpt-5-mini` route through keymaker. gpt-5 is a **reasoning
> model**, which forces two requirements that `ProprietaryLLM` and the api-mode sweep now handle:
> (1) it rejects `top_p` and non-default `temperature` → `ProprietaryLLM` sets `litellm.drop_params=True`
> so those are stripped automatically (no-op for models that do support them); (2) the token budget is
> shared with hidden reasoning tokens, so a small `--doc-max-tokens` (e.g. 512) yields **empty
> documents** — `run_sweep.sh` defaults `DOC_MAX_TOKENS=2048` in api mode (override as needed).

Docs are corrupted **in place** before the loop and tagged `distractor=true`; because the
pipeline's `initial_corpus_docs` / `current_docs` / search `corpus_candidates` share the same dict
objects, the corruption persists across rounds in every variant (search keeps each doc's original
embedding, so a distractor stays retrievable rather than dropping out). Docs that contain the gold
entity are preferred for corruption so the flip is answer-relevant. Per-question, schema-additive
`initial_distractor` record: `{fraction, mode, gold_answer, eligible, n_docs, n_corrupted, status,
distractors:[{doc_id, injected_entity, status}], ...}` — a **list**, one entry per corrupted doc.

**Diverse-aware metrics** (`hotpot_evaluation.py`, in `distractor_aggregates`; no-op without
`initial_distractor` records): per round — `gold_match_rate` (accuracy/recovery — should drop if
distractors mislead), `distractor_adoption_rate` (fraction adopting *any* seeded wrong entity),
`distinct_answers` (answer diversity — the headline signal), `offtarget_rate` (answers off-gold
*and* off-every-seeded-entity), and `distractor_retrieval_condition`. A round-0 point is included
so the shift from a wider start is visible.

## Files

| File | Role |
|---|---|
| `pipeline/misinfo.py` | `MisinfoController` (generated-stream injection) + `DistractorController` (round-0 distractors) + pure helpers (substitution, validation, realized-status, claim discovery, bridge/implied-answer, `n_to_corrupt`, `select_distractor_indices`, `choose_distinct_substitutes`) + `InjectionRecord` / `DistractorRecord` |
| `formatters.py` | freeform / untargeted / hop create-document prompts + substitute-proposal / claim-discovery / implied-answer judge prompts + `get_rewrite_distractor_conversation` (distractor rewrite) |
| `hotpot_pipeline.py` | flag wiring, seeding, both controller lifecycles (injection: prepare→make_doc_conversation→record_injection→attach; distractor: prepare_and_apply→attach) |
| `hotpot_evaluation.py` | injection-aware + distractor-aware metrics + aggregates + plot |
| `scripts/hotpot-misinfo/` | `run_misinfo.sh` (pilot), `smoke_test.sh` (now also a distractor arm), `run_variant_client.sh` / `launch_*.sh` (thread `DISTRACTOR_FRACTION`/`DISTRACTOR_MODE`), `server_answer.sh`, `server_docgen.sh`, `RUNBOOK.md` |
| `tests/test_misinfo.py` | 24 pure-Python tests (no GPU), incl. fake-LLM controller end-to-ends for both injection and distractors |

## How to run

See **`scripts/hotpot-misinfo/RUNBOOK.md`** for the full Unity walkthrough. In short:
1. Start the two vLLM servers (`server_answer.sh`, `server_docgen.sh`), export `VLLM_API_BASE` + `DOC_VLLM_API_BASE`.
2. Smoke (CPU client, 5 q × 3 rounds × 3 arms + eval + self-checks): `sbatch --export=ALL scripts/hotpot-misinfo/smoke_test.sh`.
3. Full pilot (50 q × 3 arms): `export GT_FILE=…/hotpot_dev_fullwiki_v1.json; sbatch --export=ALL scripts/hotpot-misinfo/run_misinfo.sh`.

## Status & caveats

- **M1 (final_answer; counterfactual + freeform)** and the **untargeted** control are fully
  wired and unit-tested. **Intermediate-hop (M3)** is implemented end-to-end but its bridge
  heuristic + `implied_answer` derivation need a 20–30 question manual spot-check before the
  hop metrics are trusted.
- `faithful` is a byte-identical control of the existing pipeline; all new behavior is off by
  default and the eval is a no-op without injection records.
- Verified locally: `py_compile`, `pytest` (15 passed), synthetic eval end-to-end (baseline
  parity + injection metrics), and pipeline import/flag wiring. The full run is cluster-only
  (needs vLLM servers + faiss).
