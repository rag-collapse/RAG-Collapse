# Misinformation Document Synthesis — Error-Compounding Experiment

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

## Files

| File | Role |
|---|---|
| `pipeline/misinfo.py` | `MisinfoController` + pure helpers (substitution, validation, realized-status, claim discovery, bridge/implied-answer) + `InjectionRecord` |
| `formatters.py` | freeform / untargeted / hop create-document prompts (single-line deltas from the faithful prompt) + substitute-proposal / claim-discovery / implied-answer judge prompts |
| `hotpot_pipeline.py` | flag wiring, seeding, controller lifecycle (prepare → make_doc_conversation → record_injection → attach) |
| `hotpot_evaluation.py` | injection-aware metrics + aggregates + plot |
| `scripts/hotpot-misinfo/` | `run_misinfo.sh` (pilot), `smoke_test.sh`, `server_answer.sh`, `server_docgen.sh`, `RUNBOOK.md` |
| `tests/test_misinfo.py` | 15 pure-Python tests (no GPU), incl. a fake-LLM controller end-to-end |

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
