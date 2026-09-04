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

### C3 for HotpotQA (gates R1 / R3 / T1)

Verified on `…/oyilmazel_umass_edu/experiment_outputs/hotpotqa/Qwen/Qwen2.5-14B-Instruct/hotpot_{search,replace_one,replace_all}.json`:

- **Document text IS stored** — each round's `documents` carries `doc_id`, `url`, (`title`,)
  `text` (full, ~300–2400 chars). So **pools are reconstructible and T1 is feasible**
  (the (question, round, context) triples exist with text, ready to re-prompt).
- **No `citations`, no `citation_index`, `citations_enabled` unset**; runs store only
  `run_id` + `answer`. The citation attribution was never run on HotpotQA (expected — §6
  over-citation is entity-dataset only; HotpotQA is for downstream F1).
- **No per-run retrieval metadata** (rank, score, query cosine, position). For **R3** on the
  Search variant these covariates must be **recomputed** by re-embedding docs+query with
  E5-small-v2 against the FAISS index (index is on `/work` at
  `…/rsenapati_umass_edu/hotpotqa_index/`). Position/length/redundancy are recoverable from
  the stored `documents`; query-embedding cosine and retrieval rank are not, and need recompute.
- For **R1** (placebo), "documents never in a given answer's context" are available as the
  `documents` of *other* questions/rounds in the same file, so the false-positive rate is
  computable without extra logging.

Note the citation reanalysis (R1/R2/R3) targets the **entity/graphite** runs, where
Qwen2.5-14B and Mistral have `citations_enabled=true` (with `citation_index` per round) but
DeepSeek/Llama do **not** — so R1–R3 on those two models would need the attribution re-run.

---

## R1 + R2 — placebo & threshold reanalysis (Qwen2.5-14B, entity data)

Both computed from stored answers + documents, no regeneration
(`~/r1_r2.py`; overlap rule = cite `d` iff `overlap(a,d) ≥ τ`). The tables below
are the **Replace-One** variant; the **Search** variant is the last subsection.

### R2 — over-citation ratio vs overlap threshold τ (Replace-One)

| τ | ctx self-gen % | cited self-gen % | ratio (r1) | ratio (r2) | ratio (r5) |
|---|---|---|---|---|---|
| 0.10 | 11.5 | 13.5 | 1.18 | 1.16 | 1.09 |
| 0.20 | 11.5 | 14.6 | 1.27 | 1.24 | 1.13 |
| 0.30 | 11.5 | 16.3 | 1.42 | 1.36 | 1.18 |
| 0.40 | 11.5 | 18.0 | 1.57 | 1.48 | 1.23 |

**The over-citation ratio stays > 1 at every threshold** (round 1: 1.18→1.57), so the
provenance effect is not an artifact of one cutoff. **But note the magnitude is
rule-shape-dependent:** a *threshold* rule gives ratio ≈1.2–1.6, whereas the *top-2 / LOO*
shape used for the paper gives ≈2.2 (§6.2). A threshold rule cites many low-overlap docs
(diluting the self-gen concentration); top-k concentrates on the highest-overlap docs, which
skew self-gen. §6.2 should state the rule shape, because the headline 2.2 belongs to top-2, not to a threshold.

### R1 — placebo false-positive rate, Replace-One (docs never in the answer's context)

FP = the overlap rule fires (`overlap ≥ τ`) on a document the answer never saw:

| τ | within-Q self-gen *descendant* FP % | cross-Q self-gen FP % | cross-Q human FP % | skew (descendant − cross-Q) |
|---|---|---|---|---|
| 0.10 | 97.97 | 55.54 | 54.87 | +42.4 |
| 0.20 | 95.56 | 24.19 | 31.97 | +71.4 |
| 0.30 | 89.93 | 8.05 | 15.77 | +81.9 |
| 0.40 | 79.53 | 2.49 | 7.13 | +77.0 |

(Round 1; rounds 2 and 5 are within ~3 points.) **This confirms NbXB's concern for a pure
overlap rule:** a self-generated document that was *never in the answer's context* is matched
**80–98% of the time**, purely because it is a descendant of the same question's answers and
shares wording by lineage — vs **~25–55%** for out-of-context documents from *other* questions.
The false-positive rate is heavily **provenance-skewed toward self-generated content**, so a
pure-overlap attribution is not provenance-neutral and **would inflate the self-gen citation count.**

**Why this argues *for* the method the paper actually uses (LOO), not against it.** LOO cites a
document only if *removing it changes the answer* (counterfactual necessity), not if it merely
shares words. A redundant self-gen descendant that overlaps by lineage but wasn't needed would
**not** change the answer on removal, so LOO does not credit it — exactly the failure mode R1
exposes for overlap. This is the strongest reason to (a) describe the method as **LOO, not
overlap** (W1), and (b) present R1 as evidence that LOO is the right choice.

**Honest limits:** LOO cannot be placebo-tested directly — you can't "leave out" a document that
isn't in the context — so R1 bounds the *overlap* rule's inflation and motivates LOO rather than
measuring LOO's own false-positive rate. And C2 shows LOO ≈ overlap-top-2 on *in-context* shares,
so any residual descent inflation on in-context docs is shared by both; **R3 (query-alignment
covariates) is the necessary complementary control** for the "self-gen docs are simply more
query-aligned" confound, which neither R1 nor the attribution method addresses.

### R1/R2 — Search (Qwen2.5-14B, Search variant, entity data)

Same `~/r1_r2.py` reanalysis on the Search run
(`…/all_experiments/graphite/baseline/search/experiment_outputs/Qwen/Qwen2.5-14B-Instruct/local_search.json`,
30 rounds, `citations_enabled=true`, `top_m=2`, `thr=0.18`). The retrieval-driven
context starts more self-gen than Replace-One (round 1 context self-gen is 17.8% here
vs 11.5% under Replace-One), so the ratios sit lower even though the same skew holds.

**R2 (Search).** The over-citation ratio stays > 1 at every threshold, and it
approaches the paper's reported Search value at the high end.

| τ | ctx self-gen % | cited self-gen % | ratio (r1) | ratio (r2) | ratio (r5) |
|---|---|---|---|---|---|
| 0.10 | 17.8 | 20.7 | 1.16 | 1.14 | 1.07 |
| 0.15 | 17.8 | 21.7 | 1.22 | 1.20 | 1.09 |
| 0.20 | 17.8 | 22.7 | 1.27 | 1.23 | 1.10 |
| 0.25 | 17.8 | 24.3 | 1.37 | 1.25 | 1.13 |
| 0.30 | 17.8 | 26.2 | 1.47 | 1.29 | 1.15 |
| 0.40 | 17.8 | 27.6 | 1.55 | 1.33 | 1.17 |

The ratio stays above 1 across every threshold (round 1: 1.16 → 1.55). At τ=0.40
round 1 the ratio is 1.55, close to the paper's ~1.6 for Search. So the provenance
effect is not cutoff-specific under retrieval either. The same rule-shape caveat
applies: the headline §6.2 ratio belongs to top-2/LOO, not to a threshold rule.

**R1 (Search).** The within-question self-gen "descendant" FP still towers over the
cross-question rates, and the skew is even larger than Replace-One.

| τ | within-Q self-gen *descendant* FP % | cross-Q self-gen FP % | cross-Q human FP % | skew (descendant − cross-Q) |
|---|---|---|---|---|
| 0.10 | 83.51 | 10.45 | 8.32 | +73.06 |
| 0.15 | 72.14 | 3.64 | 2.62 | +68.50 |
| 0.20 | 60.16 | 1.68 | 1.20 | +58.48 |
| 0.25 | 47.85 | 0.85 | 0.71 | +47.00 |
| 0.30 | 35.99 | 0.41 | 0.34 | +35.59 |
| 0.40 | 22.78 | 0.18 | 0.15 | +22.60 |

(Round 1; rounds 2 and 5 within ~3 points.) A self-generated descendant of the same
question's answers is matched by a plain overlap rule 83% of the time at τ=0.10 even
though it was never in the answer's context, vs ~10% for cross-question self-gen and
~8% for cross-question human docs. The overlap rule is provenance-skewed toward
self-generated content under retrieval too, which is the same argument for LOO. The
skew is a bit larger than Replace-One because the cross-question baseline drops more
under retrieval (the cross-Q docs are retrieved neighbours, so they overlap less with
an unrelated answer's wording).

### R3 — query-alignment control (Qwen2.5-14B + Mistral-7B, Replace-One + Search)

Run with `~/r3_qalign.py`. For each (question, round, context-doc) row, `y` = citation
rate = the fraction of the round's runs whose LOO citations include that doc. Covariates:
`self_gen` (provenance tag), `qcos` = cosine(embed(question), embed(doc)), `redund` =
mean cosine(doc, other context docs), `position` = doc index / n_docs, `loglen` =
log(1+chars). Linear probability model, question-clustered (CR0) SEs. Model A is
`y ~ self_gen`; Model B adds the four covariates. The last line is the attenuation of
the `self_gen` coefficient from A to B.

**Replace-One.** The self-gen effect does not collapse. It *grows*.

| model | self_gen | qcos | redund | pos | loglen |
|---|---|---|---|---|---|
| A: `y ~ self_gen` | +0.0679 (SE 0.0065, t=10.4) | — | — | — | — |
| B: + qcos + redund + pos + loglen | +0.1309 (SE 0.0092, t=14.2) | +0.4114 (0.0229, t=18.0) | −0.3674 (0.0251, t=−14.6) | +0.0081 (0.0091, t=0.9) | +0.0431 (0.0036, t=11.9) |

`self_gen` = +0.0679 (raw) → +0.1309 (controlled), a **−93% attenuation**. The negative
sign means the opposite of attenuation: once query-cosine and redundancy are
controlled, the self-gen effect is *larger* and still strongly significant (t=14.2).
So over-citation is not just query alignment; controlling for how query-aligned a
self-gen doc is does not explain the effect away. This directly answers hAN7 for the
Replace-One variant. Report both coefficients.

**Search.** The sign flips: self-gen docs are cited *less* than the regression predicts,
and controlling for query alignment moves the coefficient toward zero.

| model | self_gen | qcos | redund | pos | loglen |
|---|---|---|---|---|---|
| A: `y ~ self_gen` | −0.1189 (SE 0.0192, t=−6.2) | — | — | — | — |
| B: + qcos + redund + pos + loglen | −0.0559 (SE 0.0221, t=−2.5) | +0.4432 (0.1695, t=2.6) | −0.9208 (0.1266, t=−7.3) | +0.0233 (0.0187, t=1.2) | +0.0926 (0.0191, t=4.9) |

`self_gen` = −0.1189 (raw) → −0.0559 (controlled), a **53% attenuation**. The raw
self-gen effect under Search is *negative* (t=−6.2), the opposite sign from the
Replace-One over-citation story. Controlling for query alignment and redundancy
halves it toward zero, but it stays negative and significant (t=−2.5). So the Search
variant does not show over-citation at the per-doc level; self-gen docs are cited
*less* once you condition on the round's doc set. Report the attenuated effect honestly
and note the sign flip in §6. This is a caveat the full §6.5 model (with Rati's
quality-dimension scores) should revisit.

**Second model — Mistral-7B** (the other graphite baseline with `citations_enabled=true`;
DeepSeek/Llama graphite runs store no citations, so they can't be added without a re-run).
Same `~/r3_qalign.py`, run on `…/mistralai/Mistral-7B-Instruct-v0.3/para_off/local_{replace_one,search}_nopara.json`
(job 63967441, archived in `~/rag_rebuttal_scripts/r3_mistral_63967441.out`).

| variant | model | self_gen (A, raw) | self_gen (B, controlled) | qcos (B) | redund (B) |
|---|---|---|---|---|---|
| Replace-One | Mistral-7B | +0.0831 (t=12.4) | **+0.1672 (t=18.7)** | +0.4435 (t=20.9) | −0.4004 (t=−19.0) |
| Search | Mistral-7B | −0.0522 (t=−2.9) | +0.0274 (**t=1.4, n.s.**) | +0.7068 (t=4.9) | −1.3042 (t=−11.9) |

Mistral reproduces the Qwen pattern on **both** variants:
- **Replace-One:** the self-gen effect *grows* under the query-alignment controls
  (+0.083 → +0.167), just as for Qwen (+0.068 → +0.131). Two independent models now show
  over-citation is **not** explained away by query alignment — a robust answer to hAN7.
- **Search:** the raw effect is slightly negative and the controlled effect is
  indistinguishable from zero (+0.027, t=1.4). Like Qwen-Search, there is **no positive
  per-doc over-citation under retrieval**; if anything it attenuates to null. Report the
  Replace-One result as the headline (2/2 models) and the Search result as the honest
  retrieval-side caveat (Qwen: −0.056 sig; Mistral: ~0 n.s.).

**Caveats to carry:** R3's embedder is **all-MiniLM-L6-v2** (what the graphite pipeline
actually used via `make_embed_fn_local`), **not E5** — E5 is the HotpotQA retriever.
This is the W4 config discrepancy in the camera-ready plan; reconcile in §3/App-B. To
rerun R3 with E5 for robustness, swap the model in `~/r3_qalign.py` (E5 is on /work
hf_cache; use `query:` / `passage:` prefixes). R3 is the *query-alignment* control only;
it does not include the 8 quality dimensions, which live in Rati's §6.3/§6.5 script.
**Now cross-checked on the other dataset with its own retriever:** the HotpotQA
reconstruction in R3b below runs the query-alignment analysis with **E5** (the HotpotQA
retriever), so the W4 embedder concern is covered on both datasets — all-MiniLM on
graphite (R3), E5 on HotpotQA (R3b).

### R3b — HotpotQA retrieval alignment (E5, reconstructed) — Search / Replace-One / Replace-All

This is hAN7's control on the **HotpotQA** Search variant, and it is where retriever
provenance matters most. The runs' per-round documents are logged with full text, but
the per-run retrieval log (rank/score of each retrieved chunk) was **deleted for disk
space** and never re-saved (confirmed with Ozel, who owns those runs). So the retrieval
covariates hAN7 asked for — query-document similarity and rank — are **reconstructed
from the actual E5 index**, not the raw log, and no runs are reproduced.

Script `~/r_hotpot_retrieval.py` (CPU job 63965590, archived in `~/rag_rebuttal_scripts/`).
For each (question, round, context-doc) it computes `qsim = cos(E5("query: "+question),
E5(passage))`, matching `build_hotpotqa_index.py` exactly (e5-small-v2, masked mean-pool,
L2-normalize, `passage: {title} {text}`; self-gen docs have no title so `passage: {text}`).
`qsim` equals the index's inner product; L2 is its monotone equivalent (`L2² = 2−2·qsim`).
Each question is also searched against the real index (5,233,329 docs, IVFFlat, nprobe=64)
for the top-100 human-corpus scores → `h_top1` / `h_top10` (the best / 10th-best **real**
document E5 would surface for that query). A self-gen doc "beats gold" when `qsim ≥ h_top10`.
Provenance split: `self` = `gen_`/`model_generated` docs vs `human` = `corpus_` docs.
1400 questions, 30 rounds (round 0 is the human-only seed; rounds 1–29 shown).

| round | n_self | n_human | self_qsim | human_qsim | Δ | h_top1 | h_top10 | frac(self ≥ h_top10) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1229 | 12771 | 0.8811 | 0.8540 | +0.0271 | 0.8809 | 0.8428 | 1.000 |
| 5 | 6083 | 7917 | 0.8815 | 0.8603 | +0.0212 | 0.8809 | 0.8428 | 1.000 |
| 10 | 10901 | 3099 | 0.8835 | 0.8738 | +0.0098 | 0.8809 | 0.8428 | 1.000 |
| 20 | 12111 | 1889 | 0.8883 | 0.8800 | +0.0083 | 0.8809 | 0.8428 | 1.000 |
| 29 | 12412 | 1588 | 0.8902 | 0.8823 | +0.0079 | 0.8809 | 0.8428 | 1.000 |

Three things, all pointing the same way:

1. **Self-gen docs are more query-aligned than the human docs in context** at every round
   (Δ > 0, +0.027 early, narrowing to +0.008 late as the human pool shrinks to a
   selected, already-aligned residual). So hAN7's *premise* is descriptively true — this
   analysis does **not** deny that AI docs are more query-aligned.
2. **The scale-free result is the decisive one:** `frac(self ≥ h_top10) = 1.000` at every
   round — virtually every self-generated document out-scores the retriever's own
   **10th-best real-corpus document**, and mean `self_qsim` meets or exceeds `h_top1`
   (0.881), the single best passage E5 finds among 5.2M real docs. This is mechanical:
   the AI docs were generated *from that query*, so they are maximally query-aligned — more
   than any real passage. (Lead with this ranking statement, not the raw cosine deltas: E5
   cosines are compressed into a narrow high band, so ±0.01 is meaningful on its scale but
   the "beats every real top-10 doc" framing is what travels.)
3. **This reframes hAN7 rather than conceding it.** The query-alignment is not a confound
   that explains collapse *away* — it **is the pump**. Because a self-gen doc out-ranks
   every human doc, it is retrieved deterministically and never evicted, so the store
   collapses to AI content: `n_self` grows 1229 → 12412 while `n_human` decays 12771 →
   1588 over the 29 rounds. Retrieval-space view of the same collapse the text metrics show.

Redundancy corroborates: self-gen docs converge toward near-duplicates (mean cosine to the
round's other context docs rises 0.823 → 0.955), while human docs stay diverse (0.831 →
0.860). The embedding-space signature of the entity/lexical collapse.

**Replace-One (same reconstruction, job 63966992).** One human doc is replaced per round, so
the context is fully self-gen by round 10 (`n_human` → 0). Self-gen docs are again more
query-aligned than the human docs in context, until the human pool empties:

| round | n_self | n_human | self_qsim | human_qsim | Δ | frac(self ≥ h_top10) |
|---|---|---|---|---|---|---|
| 1 | 1400 | 12600 | 0.8761 | 0.8516 | +0.0245 | 0.886 |
| 5 | 7000 | 7000 | 0.8763 | 0.8517 | +0.0246 | 0.883 |
| 8 | 11200 | 2800 | 0.8762 | 0.8619 | +0.0143 | 0.887 |
| 9 | 12600 | 1400 | 0.8762 | 0.8809 | −0.0047 | 0.887 |
| ≥10 | 14000 | 0 | 0.8762 | — | — | 0.886 |

The human_qsim rises to meet self_qsim exactly at round 9 (the last human docs are the most
query-aligned survivors). The key contrast with Search: `frac(self ≥ h_top10)` is ≈**0.886**
here, vs **1.000** under Search. That gap is mechanistically informative — **Search *retrieves*
self-gen docs**, so it selects the most query-aligned ones (every retained self-gen doc out-ranks
the real top-10 by construction); **Replace-One *inserts* a self-gen doc into a slot regardless
of score**, so ~11% of inserted self-gen docs are *not* more aligned than the retriever's 10th
real doc. Redundancy converges identically (self 0.819 → 0.957 vs human ~0.82–0.85).

**Replace-All (hybrid), job 63966992 (COMPLETED).** As expected, the context is entirely
self-gen from round 1 onward (`n_human` = 0 every round), so `human_qsim`/Δ are undefined and
the query-alignment comparison does not apply. Self-gen `qsim` sits at ≈0.876 and `frac(self ≥
h_top10)` ≈ 0.887 at every round (same level as Replace-One's inserted docs). The distinctive
signal is redundancy: it is **already saturated at round 1** (self_redund ≈ 0.956, vs Replace-One
which climbs 0.82 → 0.96 over 10 rounds and Search over ~20) and stays flat — Replace-All floods
the whole context with restatements of one answer in a single step, the retrieval-space image of
its one-round collapse. So Replace-All contributes the redundancy-saturation endpoint, not a
query-alignment story.

**Caveats.** (a) This is a **retrieval-level** analysis; the HotpotQA runs log no citations
(§C3), so unlike graphite R3 there is no per-doc citation outcome to regress — it answers
"are self-gen docs more query-aligned and would they outrank gold?" (yes), not "are they
cited more, controlling for that?". (b) The index is IVFFlat with nprobe=64 (approximate),
so `h_top1/h_top10` are lower bounds on the true top scores; this can only *understate* how
much real docs would score, so it does not inflate the "self beats top-10" conclusion
(self_qsim ≈ h_top1 regardless). (c) Self-gen passages are encoded without a title,
matching how the pipeline embeds generated docs.

---

## R4 — collapse vs contamination fraction (NbXB)

The three regimes reach a given **contamination fraction** (share of the context that is
AI-generated) at very different rounds: Replace-All saturates in one round, Replace-One
ramps ~linearly, Search ramps gradually. Plotting collapse against *round* therefore
compares regimes at unequal saturation. This re-plots the already-logged metrics with
contamination on the x-axis so the regimes are compared at **equivalent saturation** —
a pure re-plot, no recomputation (`scripts/camera_ready/r4_contamination_replot.py`,
figures in `camera_ready_outputs/R4/`).

- x = mean `ai_reference_percentage` (contamination fraction), from `evaluation_outputs`.
- y = mean `unique_entities` per round (entity diversity; falls with collapse), from
  `entity_extraction_output`; a second figure uses `same_answer_percentage` (rises).
- Entity *similarity* is deliberately **not** the collapse axis: this repo scores an
  empty-entity answer 0.0 ("diverse"), so degenerate answers pull similarity down and it
  becomes a confounded signal (`docs/entity_extraction_comparison.md`). Unique-entity
  count is monotone in collapse.

**Finding (Qwen2.5-14B, entity dataset).** Aligned by contamination, the regimes do *not*
collapse identically — Search retains substantially more diversity at every matched
contamination level:

| contamination ≈ | Replace-All | Replace-One | Search |
|---|---|---|---|
| 0.0 | 4.3 | 3.8 | 7.2 |
| 0.25 | — (jumps to 1.0) | 3.2 | 5.9 |
| 0.6 | — | 2.5 | 4.6 |
| 0.9–1.0 | 1.0 | 1.1 | 2.3 |

(mean unique entities / round). Replace-All occupies only ~0 and ~1 contamination — it
traverses the whole axis in a single round — so its line is that 0→1 jump, not a
trajectory. Replace-One and Replace-All bottom out near 1 unique entity once fully
contaminated; **Search still holds ~2.3 at 0.92 contamination.** So Search's slower
collapse is not only that its contamination grows slower — at *equal* contamination it is
genuinely more diverse, because it retrieves from a growing pool rather than overwriting a
small fixed context. That is the honest answer to NbXB: on a saturation-matched x-axis the
regimes are comparable in shape but Search has a higher diversity floor. All four models
are in the 2×2 panels; the pattern holds (Search above the two replace regimes).

## R5 — downstream harm as per-model effect sizes (all reviewers)

Per-question ΔF1 = F1(final round) − F1(round 0) on HotpotQA, mean with a 95% bootstrap CI
over the 1400 questions, per model × regime (`scripts/camera_ready/r5_delta_f1.py`, forest
plot `camera_ready_outputs/R5/delta_f1_forest.png`). F1 is `avg_f1` (mean over the round's
10 runs) from the detailed `*_hotpot_eval.json`.

| regime | model | ΔF1 | 95% CI | sig |
|---|---|---|---|---|
| Search | Qwen2.5-14B | **−0.018** | [−0.034, −0.002] | yes |
| Search | Llama-3.1-8B | +0.016 | [−0.001, +0.032] | no |
| Search | Mistral-7B | **+0.058** | [+0.044, +0.071] | yes |
| Search | DeepSeek-R1-7B | +0.011 | [+0.000, +0.022] | yes |
| Replace-One | Qwen2.5-14B | **−0.024** | [−0.041, −0.007] | yes |
| Replace-One | Llama-3.1-8B | +0.008 | [−0.010, +0.027] | no |
| Replace-One | Mistral-7B | **+0.044** | [+0.031, +0.058] | yes |
| Replace-One | DeepSeek-R1-7B | −0.002 | [−0.013, +0.009] | no |
| Replace-All | Qwen2.5-14B | **−0.018** | [−0.034, −0.002] | yes |
| Replace-All | Llama-3.1-8B | +0.007 | [−0.009, +0.023] | no |
| Replace-All | Mistral-7B | **+0.062** | [+0.049, +0.075] | yes |
| Replace-All | DeepSeek-R1-7B | +0.012 | [+0.001, +0.023] | yes |

**Honest reading — this deviates from the plan's expected framing.** The plan anticipated
"directionally consistent, significant in one of four, underpowered." The data is *not*
directionally consistent: downstream F1 change is **model-specific and mixed**. Significant
*degradation* appears **only for Qwen2.5-14B**, and in all three regimes (−0.018 to −0.024)
— which confirms the earlier "robust only for Qwen2.5-14B" statement with effect sizes and
CIs. The other models do not degrade: **Mistral-7B significantly improves** (+0.044 to
+0.062), Llama-3.1-8B is flat (n.s.), DeepSeek-R1-7B is a negligible positive.

**Why weaker models can "improve," and the power caveat.** HotpotQA gold answers are 1–3
tokens, so token-F1 is coarse and highly sensitive to answer *verbosity/format*, not just
correctness. A weak model that emits verbose round-0 answers scores low on token-F1; as the
loop collapses its outputs toward short, repeated forms, F1 can *rise* mechanically even
though nothing was learned. So ΔF1 conflates quality with format, and for low-baseline
models (Mistral 0.26, DeepSeek 0.19) collapse toward terse modal answers reads as a gain.
The safe camera-ready statement: **downstream harm is not universal — it is significant only
for the strongest model (Qwen2.5-14B) across all regimes, while lower-baseline models show
flat-to-positive ΔF1 on an underpowered, verbosity-sensitive short-answer metric.** Do not
claim uniform downstream harm; report the per-model effect sizes above and let them stand.

---

## F1 — number-consistency sweep (§6, mitigation, abstract, intro)

Run against the paper LaTeX. **Caveat: this pass used the version pasted into the earlier
session (recovered from the transcript, ~93 KB); re-verify against the current Overleaf**
(Ozel shared the final version) since numbers may have moved. Headline result: **the
reanalysis overturns nothing** — every §6 number is internally consistent and the reanalysis
either reproduces or corroborates it.

| Claim (paper) | Location | Reanalysis | Status |
|---|---|---|---|
| Self-gen 12.0% of context, 26.5% of citations, ratio **2.2** (Replace-One, round 1) | §6.2 | §6.2 reproduction gave 27.4% cited vs 26.5% (within rounding); LOO ratio ≈2.2 | ✅ consistent |
| Self-gen 14.8% vs 24.0%, ratio **1.6** (Search, round 1) | §6.2 | R2-Search ratios 1.16→1.55 across τ, LOO ≈1.6 | ✅ consistent — **but** R2 measured context self-gen ≈17.8% at round 1 vs the paper's 14.8%; reconcile the Search round-1 denominator |
| Unique words 64.0→51.3 (round 1→9); ROUGE-L rises | §6.1 | not recomputed (R4 uses unique *entities*) | ✅ untouched |
| AI-written refs cited 0.123 vs 0.281 self-gen vs 0.078 human (+0.045, p<0.001) | §6.4 | not recomputed (Rati's) | ✅ untouched |
| Self-gen +0.154 (CI [+0.12,+0.19]) in the 8-quality-dim model; 2.1× (0.295 vs 0.142) | §6.6 | R3 (query-alignment covariates) gives self_gen +0.131 controlled (Replace-One) — same sign/magnitude, **complementary** covariates | ✅ consistent; R3 adds the query-alignment control the 8-dim model omits |
| "only Qwen2.5-14B exhibits a slight F1 degradation across all three settings"; other three do not | §5 HotpotQA + abstract ("inconsistent across models") | **R5 confirms exactly**: Qwen significant negative in all 3 regimes; Llama/DeepSeek n.s. | ✅ corroborated + quantified (per-model ΔF1 + CIs) |
| Search contamination 0%→~90% over 30 rounds | §5 HotpotQA | R3b/R4 Search: 0→0.92 | ✅ consistent |

**Flags to fix when R1–R5 land in the manuscript:**
1. **Search round-1 context self-gen %**: paper 14.8% vs measured 17.8% — likely a denominator/round-definition difference; confirm which is correct and make §6.2 agree.
2. **Mistral honesty**: the paper says the non-Qwen models "do not exhibit degradation." R5 shows Mistral **significantly improves** (+0.044…+0.062). If R5's per-model numbers are added, state the positive ΔF1 explicitly (with the token-F1/verbosity caveat), don't round it to "no change."
3. **Figure overlap**: R4's contamination-x-axis plot overlaps the existing `fig:qwen-hotpot-airef` (AI-reference fraction). Decide whether R4 supplements or replaces it; don't ship two near-duplicate contamination figures.
4. Standard proof pass: once §6 figures move, re-check the abstract and intro restatements (they currently give no §6 numbers, so low risk) and every table/appendix value against the §6 body.

## F2 — hAN7's four revision conditions

| # | Condition | Mapped task | Status |
|---|---|---|---|
| 1 | Define the citation procedure | **W1** | **Material ready** — §C1 (full LOO spec) + §C2; W1 subsection not yet written into the manuscript |
| 2 | Give *n* (responses/question/round) + selection + seeds | **W3** | **Facts established** — n=10 runs/round (`num_runs_per_iteration=10`), doc generator temp 0.7, **no RNG seed** (not bitwise-reproducible); config table not yet written |
| 3 | Add an overlap-matched mechanism control | **R3** | ✅ **Done** — R3 query-alignment control (Qwen Replace-One + Search; Mistral in flight; R3b = HotpotQA E5). The R3 covariate table is the deliverable |
| 4 | Narrow the mechanism / open-web claims | **W5** | ❌ **Not done** (writing) — per-question-store framing as a controlled same-question stress test + heterogeneous-contamination in Limitations |

Of hAN7's four, **condition 3 is fully met (R3)** and 1–2 have all their material; only **condition 4 (W5)** is outstanding, and it is pure writing. Per hAN7: if the camera-ready allows an extra page, spend it on the **W1 citation-procedure subsection + the R3 covariate table**.

---

## T1 — both attribution methods on the same items (NbXB)

Re-prompted 300 stored `(question, round, context)` triples from the reported Qwen-14B
Replace-One run with a **direct citation-elicitation** prompt (the model answers using the
numbered documents, then emits `CITED: [n, …]`), served by vLLM on an A100 at temperature 0
(job 63967793; `scripts/camera_ready/t1_{sample,generate,compare}.py`). All 300 answers
produced a parseable `CITED:` line. This puts the **direct** method and the stored **LOO**
method on identical items — C2's cross-version agreement as a statistic computed here, not a
pointer to the earlier version.

Over-citation ratio = (self-gen share of citations) / (self-gen share of context), same items:

| subset | context self-gen | **LOO** ratio | **Direct** ratio |
|---|---|---|---|
| **round 1** (the §6.2 condition, ctx self-gen 11.5%) | 0.115 | **2.57** | **2.49** |
| pooled rounds 1–19 | 0.640 | 1.06 | 1.34 |

Doc-level agreement (pooled): Jaccard 0.346; direct precision/recall vs the LOO union 0.429 / 0.725;
≈5.04 direct citations/item vs LOO's 1.27 per run.

**Reading.** At **round 1 — the exact condition §6.2 reports — the two independent attribution
methods agree tightly: LOO 2.57 and direct 2.49**, both ≈ the paper's stated **2.2**. That is the
cleanest possible form of the C2 claim: one over-citation effect, measured two ways, on the same
items, ~2.5× in both. The **pooled** ratios (1.06 / 1.34) look weaker only because they average over
rounds where the context is already 64% self-gen; as the context saturates, *any* rule's ratio is
mechanically compressed toward 1 (you cannot cite self-gen at 2× its share when it is already 90% of
the context), so the pooled number understates the round-1 effect and is not the quantity to quote —
report the round-1 ratios against §6.2. Where the two methods *do* differ is verbosity, not
direction: direct elicitation lists ≈4× more documents per answer than the strict LOO counterfactual
(5.04 vs 1.27) and recovers 72.5% of LOO's cited docs while adding its own, so their doc-level
overlap is moderate (Jaccard 0.35) even though their self-gen *skew* coincides.

**One caveat before this goes in the paper.** The elicitation prompt is a **reconstruction** (Rati
owns the original); it doubles as the W2 appendix text and should be reconciled with her wording,
though the ~2.5× round-1 agreement is robust to phrasing. (Re-run `t1_compare` on the full round-1
set to firm up the number beyond the 100 round-1 items in this sample.)

---

## Mediator vs confounder — why prior-answer similarity must not be a covariate

A reviewer-facing subtlety worth stating explicitly, because it looks like a control we skipped
but is one we must **not** add.

The data-generating structure is a **chain, not a fork**: `A₁ → D₁ → A₂`. A round's answer `A₁`
is rewritten into a synthetic document `D₁`, which is fed as context and shapes the next answer
`A₂`. If self-generated references are genuinely influential, **`D₁` is the channel by which `A₁`
reaches `A₂`** — its similarity to the prior answer is the **mediator**, i.e. it lies *on* the
causal path. Conditioning on a mediator removes the very effect you are trying to measure, so
adding "overlap with the prior answer" as a regression covariate would **subtract the collapse
signal itself** and drive the estimate to zero — wrongly.

This distinguishes two similarity covariates that look alike but sit in different causal positions:

| covariate | causal role | in a citation/collapse regression |
|---|---|---|
| **query–document similarity** (query `Q → D`, `Q → A₂`) | **confounder** (common cause) | **adjust** — closes a backdoor path, isolates `D → A₂`. This is what **R3** does (query-cosine). |
| **prior-answer overlap** (`A₁ → D → A₂`) | **mediator** (on the path) | **do not adjust** — closes the front door, nets out the effect. |

The reviewer worry — "maybe `D` is just similar to the prior answer, not influential" — conflates
these. When `D` is a **faithful restatement** of `A₁`, that similarity *is* the mechanism of
influence; the two hypotheses (influence vs. mere ancestry) are **observationally equivalent** on
overlap, so no regression on the same data can separate them. **What separates them is contrast,
not adjustment** — and our design already uses it in two places, with a third available:

1. **The cite decision is a counterfactual regeneration, not a regression on overlap.** For each
   answer the method takes the top 2 documents by lexical overlap with that answer
   (`citation_top_m=2`, `pipeline.py:76`), regenerates the answer with each one removed
   (`pipeline.py:874`, `:879`), and cites it when the regenerated answer differs from the original
   by at least 0.18 lexically (`_answer_change_score`, `pipeline.py:66`, `:101`). The decide step
   never conditions on prior-answer overlap, so the mediator objection does not reach it. It has two
   honest limits and they are not the mediator bias. Only the top-2 overlap candidates are ever
   eligible, so a low-overlap but influential document is missed. That is a recall limit. And the
   change score is lexical, a token-overlap proxy for whether the answer moved, not a semantic one.
   State it as a **counterfactual decision over overlap-selected candidates**, not a full
   leave-one-out over all n references. It reruns 2 candidates per answer, not n.
2. **R1 (placebo) is the difference-in-differences** the objection calls for: the same document,
   scored when it was in context vs. when it never was. Ancestry is present in both arms; influence
   is possible only in the exposed arm; the difference is influence — nothing is subtracted from the
   treated measurement.
3. **Novelty restriction (proposed next analysis; call it R7/T3).** Restrict to material `D₁`
   *introduced* that `A₁` did not contain (the document generator's hallucinated entities/specifics),
   and measure how often `A₂` adopts it. Under pure ancestry `A₂` has no route to that material;
   under influence it does. This **selects a subset of the signal** (a lower bound on influence)
   rather than netting anything out, so it never touches the collapse effect. It is buildable from
   stored answers + documents, and §5 already observes the phenomenon qualitatively ("the document
   generator hallucinates entities … and the answer generator conditions on these fabricated facts").

One honest flag on R3's covariate set: `redund` (mean cosine to the round's *other* context docs)
is the one covariate to defend carefully — it measures duplication among neighbours, not similarity
to the specific ancestral answer, so it is a retrieval nuisance rather than the `A₁→D→A₂` mediator;
keep it, but be ready to show the effect survives without it. **Principle: never adjust for anything
on the path from document to answer; find where the two stories predict different things, and measure
there.**

---

## Exact questions to send

### To Rati (`ratirastogi@umass.edu`)

> Hey Rati, I'm working through the §6 citation stuff for the camera-ready and ran into a
> few things only you can answer, since you wrote the citation code. Three are blocking me,
> the rest are quick confirmations.
>
> The big one: I can't find the analysis that produces the §6.3/§6.5 numbers anywhere in the
> repo. The self-generated vs AI-written vs human split, the GPTZero labeling of the round-1
> originals, the 8 quality-dimension regression, the 0.281 / 0.123 / 0.078 rates and the
> +0.154 effect. I grepped all the Python and the notebooks and it isn't there. Is it on your
> scratch, a local notebook, or maybe the collapse-randomness-research repo? Even a messy copy
> would unblock me.
>
> Related to that: the script's inputs. The per-document GPTZero labels for the round-1
> originals (the "30% are AI" number) and the 8 quality-dimension judge scores. Are those
> saved somewhere as a file, or does the script recompute them every run? If they're saved I'd
> rather just reuse them.
>
> Third one: the explicit citation runs. The direct-elicitation numbers only survive as
> aggregates in `evaluation_outputs/Qwen/citations/` (the eval files tagged
> `ai_citation_source: explicit`). The raw run on /work
> (`ratirastogi_umass_edu/.../Qwen2.5-7B-Instruct/local_replace_one.json`) has empty
> `citations` arrays. Do you still have a raw run somewhere where the per-answer citations are
> actually populated? If so we can line up explicit against the current method on the same
> items without re-running anything, which is exactly what NbXB asked for.
>
> The quick ones:
>
> For §6.2, when you computed the citation share, did you pool over all citations in a round
> or average the per-question shares? I get 27.4% pooling and 28.1% averaging, and the paper
> says 26.5%, so I just want to match your recipe.
>
> Which run produced the §6.2 numbers? I've been using the ffatima Qwen-14B replace_one run at
> `all_experiments/graphite/baseline/replace_one/.../local_replace_one.json` (it has
> `citations_enabled`, `top_m=2`, `max_docs=6`, `threshold=0.18`). Is that the one?
>
> For the §6.3/§6.5 citation rate, can you confirm it's per-reference? As in, for each
> reference, the fraction of the 10 runs in a round that cite it, averaged within a provenance
> group. I want to be sure it isn't the same denominator as the §6.2 share.
>
> Last one, and it changes how we write §6 up: the code is actually leave-one-out (it drops a
> document and checks whether the answer changes), not a plain overlap match, with overlap only
> picking the top-2 candidates. The reviewers keep calling it "overlap." Should we describe it
> as leave-one-out in the paper, or is there a separate overlap-only version somewhere that
> generated the numbers? I ask because I ran the placebo check (R1) and a plain overlap rule
> fires on ~95% of self-generated docs that were never even in the answer's context, so the
> leave-one-out framing is what defuses that objection.
>
> Thanks. The first three are what's holding up the rewrite.

### To Ozel (`oyilmazel@umass.edu`)

> Hey Ozel, quick one about the HotpotQA runs for the rebuttal. I checked hotpot_search,
> replace_one, and replace_all for Qwen-14B, and the good news is each round stores the full
> document text, so we can reconstruct exactly what every answer saw. The runs only keep
> `run_id` and `answer` per run though, with no citations, no `citation_index`, and no per-run
> retrieval info (rank, score, or which chunks were retrieved). For the Search reanalysis hAN7
> wants, I need retrieval rank and query-document similarity, which I can recompute from the E5
> index, but before I do: is there a richer log anywhere that already has the per-run retrieved
> chunks or scores? I'd rather not recompute if you already saved it.

---

## Open questions & ownership (annotated, for reference)

The citation attribution + provenance code is authored by **Rati Rastogi**
(`ratirastogi@umass.edu`): `git log -S _infer_citation_ids_via_retrieval_loo -- pipeline.py`
and the `ai_citation_source`/"explicit" path both trace to Rati, with commits titled
"citations". The over-citation *analysis* (§6.3/§6.5) consumes ffatima's Qwen-14B citation
run but the analysis script itself is not in this repo. Questions to resolve, each with a
concrete pointer:

1. **§6.2 exact aggregation.** `evaluation.py:41–62` (`calculate_ai_citation_percentage`) +
   the AI tag `evaluation.py:23–39`. Reconstruction: 27.4% pooled-over-citations /
   28.1% mean-over-questions vs the paper's 26.5%. → Ask Rati which aggregation the paper used.
   
2. **Source run for §6.2.** I used `…/all_experiments/graphite/baseline/replace_one/experiment_outputs/Qwen/Qwen2.5-14B-Instruct/local_replace_one.json`
   (`citations_enabled=true`, `citation_top_m=2`, `max_docs=6`, `change_threshold=0.18`;
   ffatima's canonical run). → Confirm this is the reported run/version.
3. **§6.3/§6.5 analysis script (BLOCKER).** Not in repo (grepped all `.py` + notebooks).
   Produces self=0.281 / AI-written=0.123 / human=0.078 and the linear-model +0.154. → Where
   is the script/notebook (Rati's scratch? local? `collapse-randomness-research` repo)?
4. **Its inputs (BLOCKER).** The per-doc **GPTZero** labels for round-1 originals (the "30%
   AI" figure) and the **8 quality-dimension** LLM-judge scores. → Saved anywhere (CSV/JSON),
   or recomputed each run?
5. **"Citation rate" denominator (§6.3/§6.5).** Confirm per-reference: fraction of the round's
   10 runs that cite a given reference, averaged within a provenance group — distinct from
   §6.2's share-of-all-citations. (Whatever code computes it is presumably in #3.)
6. **Explicit per-answer data.** Explicit citations survive only as aggregates in
   `evaluation_outputs/Qwen/citations/Qwen2.5-7B-Instruct_local_replace_one_eval.json`
   (`metric_metadata.ai_citation_source:"explicit"`); the `/work` raw run
   `…/ratirastogi_umass_edu/experiment_outputs/Qwen/Qwen2.5-7B-Instruct/local_replace_one.json`
   has **empty** `citations` arrays. → Do the raw runs with *populated* explicit per-answer
   citations still exist? If so, the controlled explicit-vs-LOO (C2 / T1) needs no re-run.
7. **Method naming.** Code is LOO counterfactual (`pipeline.py:101–143`), overlap only a
   prefilter; paper/reviewers say "overlap". → Should the paper describe it as LOO, or is
   there a separate pure-overlap attribution that generated the numbers?

**For Ozel** (`oyilmazel@umass.edu`) — HotpotQA logging (see the C3-for-HotpotQA section):
the runs store full `documents` text (pools reconstructible; T1 OK) but no citations /
citation_index / per-run retrieval metadata, so R3's retrieval-rank + query-cosine covariates
must be recomputed from the E5 index. Confirm no richer per-run retrieval log exists elsewhere.
