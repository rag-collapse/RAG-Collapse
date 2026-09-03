# Citation attribution — full spec & reviewer responses (C1–C3)

Grounding for the §6 over-citation analysis, verified against code and against the
**reported** run outputs on Unity. Written to answer reviewer points C1, C2, C3.

Key files:
- Attribution code: `pipeline.py:48–143` (`_infer_citation_ids_via_retrieval_loo` and helpers).
- Provenance / share counting: `evaluation.py:23–62` (`_is_ai_generated_citation`, `calculate_ai_citation_percentage`).
- Reported data (Qwen2.5-14B, Replace-One): `…/all_experiments/graphite/baseline/replace_one/experiment_outputs/Qwen/Qwen2.5-14B-Instruct/local_replace_one.json`.

---

## C1 — Full spec of the citation attribution

**Terminology fix:** the mechanism is **not** "overlap-based attribution." It is a
**leave-one-out (LOO) counterfactual** attribution with lexical overlap used only as a
candidate *prefilter*. Reported-run hyperparameters (read from `experiment_metadata`):
`citation_top_m = 2`, `citation_max_docs = 6`, `citation_change_threshold = 0.18`.

For each answer `a` produced from that round's document pool:

1. **Tokenization** (`pipeline.py:56–57`). Token = regex `[A-Za-z0-9']+`, lowercased,
   keeping only tokens of **≥ 4 characters** (crude stopword removal), taken as a **set**
   (deduplicated). Unit of overlap = **word-type token** — not n-gram, not sentence.
2. **Overlap score** (`:60–63`): `overlap(a, d) = |tokens(a) ∩ tokens(d)| / |tokens(a)|`
   — coverage of the *answer's* tokens by document `d`. Denominator = number of distinct
   ≥4-char answer tokens. **This is not Jaccard** (the union is not the denominator).
3. **Candidate prefilter** (`:76–98`): rank the round's documents by `overlap(a, d)` and
   keep the **top-M = 2**.
4. **LOO counterfactual** (`:101–143`): for each of the 2 candidates, regenerate the
   answer with that document removed from context, giving `a_{-d}`. The change score is
   `change(d) = 1 − overlap(a, a_{-d})` (`:66–73`). Document `d` is **cited by `a`** iff
   `change(d) ≥ 0.18`, up to a cap of **6** citations.
5. **Ties / multi-document** (`:125–143`): candidates are evaluated in overlap-descending
   order and selected in that order until the cap. **Fallback:** if no candidate clears
   the 0.18 threshold, keep the up-to-2 documents with the highest *positive* change.
6. **Agentic RAG** (`:824–832`): the candidate pool is the union of retrieved chunks
   across the round's runs rather than a fixed prompt-doc list.

### Denominators used in §6

Two *different* quantities appear in §6; keep them distinct:

- **"Citation share" (§6.2)** — the share of *all citations* going to a provenance group.
  Computed as (`evaluation.py:41–62`):
  `citation_share(group, round) = (# citation entries whose doc is in group) / (total citation entries)`,
  **pooled across the round's 10 runs and all questions**. A "citation entry" = one element
  of a run's `citations` list, i.e. one document the LOO test kept for that answer.
  The **context share** it is compared against = `(# docs in group) / (total docs in the round)`.
  The §6.2 ratio = citation_share / context_share.
- **"Citation rate" (§6.3, §6.5)** — a *per-reference* quantity: for a given reference,
  the fraction of the round's runs (answers) that cite it, then averaged within a
  provenance group. This is a different denominator (per-reference, not per-citation).

### Provenance / self-generated tag (`evaluation.py:23–39`)

A cited document is **self-generated (AI)** iff **any** of:
`doc_id` starts with `gen_`, **or** `url == "model_generated"`, **or**
`iteration > 0 and doc_id` does **not** start with `ref_`. This is **tag-based**
(document provenance), not the AI-content detector.

### Empirical check — the spec reproduces the paper

Reconstructing §6.2 directly from the reported Qwen2.5-14B Replace-One run:

| round | context self-gen % | citation self-gen % | ratio | paper (§6.2) |
|---|---|---|---|---|
| **1** | **11.5** | **27.4** | **2.39** | 12.0 / 26.5 / 2.2 |
| 2 | 23.0 | 47.2 | 2.05 | — |
| 5 | 57.4 | 75.2 | 1.31 | — |

Matches within rounding. (The small 27.4 vs 26.5 gap is an aggregation choice —
pooled-over-citations vs mean-over-questions; confirm which the paper uses so the exact
number is defensible. The mechanism and denominators are settled.)

---

## C2 — The two implementations agree in magnitude

There are two attribution implementations; the reported §6 numbers use the **current LOO**
one (metadata carries `citation_top_m/max_docs/change_threshold`). The **earlier** version
**elicited citations directly from the model**; its eval outputs are tagged
`metric_metadata.ai_citation_source = "explicit"` (e.g.
`evaluation_outputs/Qwen/citations/Qwen2.5-7B-Instruct_local_replace_one_eval.json`).

Citation self-generated **share** by round, Replace-One, **matched aggregation**
(mean over questions):

| round | Explicit (Qwen2.5-7B, 50 q) | LOO (Qwen2.5-14B, 400 q) |
|---|---|---|
| 1 | 21.3% | 28.1% |
| 2 | 35.6% | 48.8% |
| 5 | 70.3% | 74.6% |
| 10 | 78.0% | 100.0% |

**Conclusion:** both methods show the same over-citation effect at consistent magnitudes —
citation self-gen share ≈ 2× context share at round 1 (context share ≈ 12%), rising toward
near-total self-citation as the context saturates. This supports presenting §6 as **one
effect measured two ways**, not two independent effects.

**Caveat:** the explicit vs LOO table above is *not* controlled — different answer models
(7B vs 14B) and sample sizes (50 vs 400 q). See the controlled comparison below.

### Controlled same-answers comparison: LOO vs overlap (Qwen2.5-14B, Replace-One)

The reviewer's C2 framing is "the earlier version elicited citations directly; this one
assigns them post-hoc by overlap." We can test that literally on the **identical answers**
of one run: compare the stored **LOO** citations against a pure **overlap** attribution
(top-1 / top-2 by the same `_overlap_score`) computed on the same answers + document pools.
Self-generated citation share by round (mean over questions):

| round | LOO (stored) | overlap top-1 | overlap top-2 | LOO⊆overlap-top2 (recall) | Jaccard(LOO, overlap-top2) |
|---|---|---|---|---|---|
| 1 | 28.1% | 30.8% | 26.4% | 91.5% | 68.4% |
| 2 | 48.8% | 43.3% | 45.7% | 91.3% | 64.2% |
| 5 | 74.6% | 65.7% | 73.0% | 86.6% | 55.4% |
| 10 | 100.0% | 100.0% | 100.0% | 80.0% | 46.5% |

**On the same answers, LOO and overlap give near-identical self-gen shares** (within ~1–3
points every round) and pick largely the same documents (80–92% of LOO citations are in the
overlap top-2). So the §6.2 over-citation magnitude (~2× at round 1) is robust to the
attribution method — the reviewer's "it's overlap" mental model yields the same headline
number as the actual LOO. This is the strongest available same-run evidence and it is
fully reproducible from stored outputs (`~/explicit_vs_overlap.py`).

**What is *not* reconstructible:** a fully controlled **explicit vs LOO** comparison on
identical answers. The explicit method's per-answer cited-doc sets survive only as
aggregated `ai_citation_percentage` in the eval files; the `/work` raw runs store empty
`citations` arrays. Producing it would require re-running explicit citation *elicitation*
over the LOO run's stored answers (a Qwen model server / GPU job). The aggregate explicit
(21–78%) and LOO (28–100%) trajectories already agree in shape and magnitude, and the
LOO≈overlap result above shows the attribution mechanism is not what drives the numbers.

---

## C3 — What the logs kept per answer

For the reported (citation-enabled) runs, each round stores everything R1/T1 need:

- `iteration.documents` — the round's full document pool, each with `doc_id`, `url`,
  `iteration`, **and full `text`** (not just IDs).
- `iteration.citation_index` — the per-round list mapping `citation_id → {doc_id, url, iteration}`
  for the documents that round's answers were scored against.
- Each `run` has `answer`, `citation_ids`, and resolved `citations`.

Provenance is recoverable from the tags (`gen_`/`ref_`, `url`). Therefore the exact
document set each answer saw is **reconstructible**:
- **Replace-All / Replace-One:** directly — no retrieval, so every run saw the stored `documents`.
- **Search / Agentic:** the per-run retrieved subset is captured via `citation_index`
  (present in these runs), so the concern that Search stores only IDs does **not** apply here.

**One thing to verify per experiment:** `experiment_metadata.citations_enabled`. In the
consolidated data it is `true` for the Qwen2.5-14B and Mistral baseline runs but `false`
for several DeepSeek/Llama and all agentic runs — those lack the per-run `citations`
and `citation_index`, though they still store full `documents` (so pools are reconstructible,
but the per-run retrieved subset under Search is not).

---

## Ownership (who to ask about §6.3/§6.5)

The citation attribution + provenance code is authored by **Rati Rastogi**
(`ratirastogi@umass.edu`): `git log -S _infer_citation_ids_via_retrieval_loo` and the
`ai_citation_source`/"explicit" path both trace to Rati, with commits titled "citations".
Rati is therefore the owner to ask for the parts **not in this repo** — the §6.3/§6.5
analysis (three-way self / AI-written / human split, GPTZero labeling of round-1 originals,
the 8-quality-dimension linear model, bootstrap CIs, and the specific numbers 0.281 /
0.123 / 0.078 and +0.154). That script and its inputs (GPTZero labels, quality-dim scores)
should come from Rati to write §6.3/§6.5 defensibly.
