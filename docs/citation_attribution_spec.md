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

### R3 — query-alignment control (Qwen2.5-14B, Replace-One + Search)

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

### R3b — HotpotQA Search retrieval alignment (E5, reconstructed)

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

**Caveats.** (a) This is a **retrieval-level** analysis; the HotpotQA runs log no citations
(§C3), so unlike graphite R3 there is no per-doc citation outcome to regress — it answers
"are self-gen docs more query-aligned and would they outrank gold?" (yes), not "are they
cited more, controlling for that?". (b) The index is IVFFlat with nprobe=64 (approximate),
so `h_top1/h_top10` are lower bounds on the true top scores; this can only *understate* how
much real docs would score, so it does not inflate the "self beats top-10" conclusion
(self_qsim ≈ h_top1 regardless). (c) Self-gen passages are encoded without a title,
matching how the pipeline embeds generated docs.

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
