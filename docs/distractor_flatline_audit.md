# Why `gold_match` flatlines across the distractor sweep: code audit

**Question audited:** in `diverse_synth` / `equal_diverse_synth`, accuracy (`gold_match`) drops
once *any* distractor is present and then stays flat no matter how high the fraction goes.

**Scope:** `pipeline/misinfo.py` (`DistractorController`), `hotpot_pipeline.py` (injection hooks),
`base_hotpotqa_distractors/compare_sweep.py` (metric extraction), `hotpot_evaluation.py`
(the *other* implementation of the same metrics), `base_hotpotqa_distractors/make_1400q_figures.py`.

**Evidence base:** the local 400q pilot arms (`distractor_sweep/`, `equal_distractor_sweep/`),
which still contain raw per-run answers and `initial_distractor` records, plus the 24 fetched
1400q summaries in `1400q_summaries/`.

**Verdict:** the flatline is **part measurement bug, part design**. Fixing the metrics makes the
DeepSeek/Llama panels meaningful and cleans up the adoption curve, but it does *not* create a
dose-response for `diverse_synth`. It inverts it. Two design decisions (§5, §6) cap the dose.

---

## 0. The injection itself is NOT broken

Verified end-to-end on the pilot arms: for every selected slot, the recorded round-0 document
actually asserts its injected wrong entity.

```
base_search_f0.3   3 slots/question    354/354 slots assert the entity
base_search_f0.7   7 slots/question    749/749 slots assert the entity
base_hybrid_f0.3   3 slots/question    363/363
base_hybrid_f0.7   7 slots/question    777/777
```

So `DistractorController.prepare_and_apply` → in-place `s.current_docs` mutation → the
`documents` recorded in iteration 0 all line up. The dose *is* delivered (3 vs 5 vs 7 docs).
Don't look for the bug here.

---

## Bugs in the extraction code (`base_hotpotqa_distractors/compare_sweep.py`)

### 1. `seeded` is a UNION across arms: `compare_sweep.py:60–67`

```python
for p in args.runs:
    d = json.load(open(p))
    arms[_fraction_of(p)] = d
    for q in d.get("questions", []):
        rec = q.get("initial_distractor")
        if rec and rec.get("eligible"):
            cid = _qid(q)
            eligible.add(cid)
            ents = {normalize(x["injected_entity"]) for x in rec.get("distractors", [])
                    if x.get("injected_entity")}
            seeded.setdefault(cid, set()).update(ents)   # <-- union over ALL arms
```

Every arm's `distractor_adoption` and `offtarget` are scored against the union of *every* arm's
injected entities. The 1400q runs load a frozen pool (`--distractor-docs-file`) and slot selection
is a seeded prefix, so the arms are **exactly nested** → the union equals the largest arm's set.
Result: f=0.3 (3 seeded entities) is scored against f=0.7's 7 entities.

Two visible consequences:

- the adoption dose curve is mechanically flattened (low arms get credit for entities they never seeded);
- the **f=0 baseline reports non-zero adoption** (`a0 = 0.004–0.005` in every 1400q summary) despite
  seeding nothing at all.

**Fix:** keep per-arm seeded sets; score each arm against its own entities only.

### 2. `gold_match` is exact equality on the whole answer: `compare_sweep.py:87`

```python
gm += sum(a == g for a in abr) / len(abr)
```

where `a = normalize(run["answer"])` (the **entire** answer text) and `g = normalize(gold)` (a short
entity). No answer extraction, no containment. Consequences:

- **DeepSeek-R1-Distill emits reasoning**, so its answer never equals the bare gold string:
  `gold_match ≈ 0.004–0.013` at *every* arm, and `distinct_answers ≈ 9.4–9.7 / 10`, i.e. already at
  the ceiling so it cannot rise with dose either. **The entire DeepSeek row of every figure is an
  artifact, not a result.**
- Even Qwen is understated. Recomputing round 0 on the pilot with `contains_entity`:

  | arm | `gm` as coded (exact) | `gm` with containment |
  |---|---|---|
  | f=0   | 0.499 | 0.676 |
  | f=0.3 | 0.153 | 0.207 |
  | f=0.5 | 0.202 | 0.282 |
  | f=0.7 | 0.231 | 0.322 |

- All the formatting noise lands in `offtarget`, inflating it.

**Note:** `hotpot_evaluation.py:186–225` (`_distractor_metrics_for_iteration`) already computes these
same four metrics **correctly**, using `contains_entity` and per-question seeded entities.
`compare_sweep.py` is a second, inconsistent implementation of the same thing, and it is the one
feeding the figures.

**Fix:** delete the metric core in `compare_sweep.py` and call
`hotpot_evaluation._distractor_metrics_for_iteration`. That kills bugs 1–3 and the duplication.

### 3. The cohort is the UNION of per-arm eligibility, not the intersection: `compare_sweep.py:62–64`

`eligible` accumulates any qid that *any* arm marked eligible. In the pilot the per-arm
`no_valid_substitute` count differs (13 / 12 / 15 of 150, because the gpt-5-mini proposal call is
sampled per arm), so ~8% of the cohort in any given arm seeded **nothing** yet is still scored
against the union's entities, a baseline run counted as a treatment run.

```
search        union cohort 130   intersection 110
hybrid        union cohort 130   intersection 109
replace_one   union cohort 128   intersection 109
```

The frozen pool used at 1400q mostly hides this (eligibility is set before the cache lookup, so it
is uniform across arms), but the fix is the same and it matters for the 400q pilot figures.

**Fix:** intersect. Also treat `status == "no_docs_in_cache"` (n_corrupted 0 but `eligible=True` in
load mode, `misinfo.py:852`) as ineligible.

### 4. `_fraction_of` fallback crashes later: `compare_sweep.py:27–33`

On regex miss it returns the **basename**, which then reaches
`sorted(arms, key=lambda f: float(f))` at line 70 → `ValueError`. Should raise a clear error at
parse time rather than deep in the sort.

### 5. Bonus, in `hotpot_evaluation.py:203`

```python
flagged = any(d.get("distractor") for d in docs)
```

With `--initial-docs native_distractor`, `native_context_docs` tags the 8 native TF-IDF paragraphs
`distractor=True` (`misinfo.py:201`). So `distractor_retrieval_condition` is **1.0 for every arm,
including f=0**. It should key off `dist_ids & doc_ids` (the *synthesized* slots) only.

---

## Why fixing those does not produce the curve

Recomputed the pilot with all three fixes applied: containment matching, per-arm seeded sets,
intersected cohort. The trend does not flatten; it **inverts**:

```
=== search        (cohort = 110, intersected)
  arm     r0 gold   r0 adopt      rF gold   rF adopt
  f=0       0.688     0.000         0.661     0.000
  f=0.3     0.186     0.702         0.223     0.675
  f=0.5     0.255     0.637         0.270     0.590
  f=0.7     0.315     0.598         0.237     0.633

=== hybrid        (cohort = 109)
  f=0       0.672     0.000         0.579     0.000
  f=0.3     0.184     0.705         0.191     0.569
  f=0.5     0.226     0.656         0.202     0.573
  f=0.7     0.259     0.666         0.266     0.539

=== replace_one   (cohort = 109)
  f=0       0.692     0.000         0.642     0.000
  f=0.3     0.182     0.734         0.181     0.733
  f=0.5     0.239     0.663         0.259     0.663
  f=0.7     0.241     0.618         0.285     0.594
```

Accuracy **rises** and adoption **falls** as the fraction goes up. Two design reasons:

### 6. `diverse_synth` raises the number of *competing* falsehoods, never the support for any one

`_apply_diverse_synth` (`misinfo.py:890–964`) gives each corrupted slot its **own distinct** wrong
answer, one document each. So as the fraction rises:

- number of distinct wrong answers: 3 → 5 → 7
- support per wrong answer: **1 document, constant**
- support for gold: **2 native gold paragraphs, constant**

At f=0.7 the gold answer is the *best-supported single answer* in the context. That is why the extra
distractors push mass into `offtarget` rather than `adoption`, clearest in Mistral, where round-0
adoption *decreases* monotonically as offtarget climbs:

```
mistral-7b / search / diverse_synth   a0:  0.233  (f=0.3)  →  0.167  (f=0.5)  →  0.136  (f=0.7)
```

It is also why `equal_diverse_synth`, which varies the **count of 2-document topics**, holding
per-topic support fixed, is the one axis that *does* show a monotone dose:

```
qwen2.5-14b / search / equal_diverse_synth
  T=1  g0=0.164  a0=0.551
  T=2  g0=0.147  a0=0.587
  T=3  g0=0.132  a0=0.601
  T=4  g0=0.137  a0=0.604
```

The headline this supports is: **dose-response tracks support per falsehood, not number of
falsehoods.** The `diverse_synth` fraction axis is the wrong knob for a monotone accuracy curve.

### 7. `n_to_corrupt` denominator includes the gold docs it can never corrupt

`misinfo.py:269–274`:

```python
n = max(0, min(k, int(round(fraction * k))))      # k = len(docs) = 10
```

but `--distractor-avoid-gold` (set in every sweep script) restricts `select_distractor_indices` to
the **8 non-gold slots**. So realized corruption is `min(round(10·f), 8)`:

| nominal f | slots corrupted | fraction of *corruptible* slots |
|---|---|---|
| 0.3 | 3 | 0.375 |
| 0.5 | 5 | 0.625 |
| 0.7 | 7 | 0.875 |
| 0.8 | 8 | 1.000 |
| 0.9 | 8 | 1.000 |
| 1.0 | 8 | 1.000 |

Two problems: the x-axis label "distractor fraction" overstates the realized dose, and **f ≥ 0.8
would be a literal hard flatline**, three identical arms. Confirmed in the pilot counts
(`f=0.7 → n_corrupted {7: 107, 6: 6, …}`, `T=4 → {8: 93, 6: 19, …}`; the sub-maximal values are
questions where a native distractor paragraph happens to contain the gold string and is therefore
excluded by `_is_gold_doc`).

Also: because `avoid_gold=1`, the 2 gold paragraphs are **never displaced in round 0 in any arm**.
Accuracy has a floor by construction. (They persist across all rounds in `search` and most rounds in
`replace_one`; in `hybrid`/`replace_all` with `--num-synth-docs 10 --num-db-docs 0` the context is
fully replaced after round 0, so there the pinning only affects round 0, which is where the whole
dose lands anyway.)

**Fix:** compute `n` against the corruptible-slot count, and relabel the axis to the realized
fraction. If you want accuracy to actually degrade with dose, you also need an arm that displaces
gold evidence (`--distractor-avoid-gold` off, or a variant that drops gold paragraphs).

---

## Bug in the figure code (`base_hotpotqa_distractors/make_1400q_figures.py`)

### 8. `sharey=True` with DeepSeek in the panel: lines 155, 187 (and 95, 121)

For `distinct_answers`, DeepSeek sits at ~9.7 and Qwen at ~2.0. The shared y-axis squashes every
other model into a flat line at the bottom, a **visual** flatline stacked on top of the real one,
and the `axhline(1.0)` collapse reference ends up pinned to the axis floor. Affects
`dose_all_models` and `traj_all_models`, and the left/right `sharey` in `regen_model`.

**Fix:** drop `sharey` for `distinct_answers` (or normalize by `num_runs`). This becomes much less
severe once bug 2 is fixed, since DeepSeek's 9.7 is itself an artifact of unextracted answers.

---

## Suggested fix order

1. **`compare_sweep.py`**: replace its metric core with a call to
   `hotpot_evaluation._distractor_metrics_for_iteration`; per-arm seeded sets; intersected cohort;
   hard error in `_fraction_of`. *(Bugs 1–4.)* This is the one that changes published numbers.
2. **`hotpot_evaluation.py:203`**: `distractor_retrieval_condition` off `dist_ids & doc_ids` only. *(Bug 5.)*
3. **`misinfo.py:269`**: `n_to_corrupt` against corruptible slots; relabel the figure x-axis. *(Bug 7.)*
4. **`make_1400q_figures.py`**: unshare y for `distinct_answers`. *(Bug 8.)*
5. **Framing**: stop presenting the `diverse_synth` fraction axis as a dose-response on accuracy.
   Lead with `equal_diverse_synth` (§6), and if a monotone accuracy curve is wanted, add an arm that
   displaces the gold paragraphs.

---

## Post-fix results (1400q, recomputed 2026-07-29)

`compare_sweep.py` was patched (containment matching, per-arm seeded sets, intersected cohort, hard
`_fraction_of`), commit `165177c`. All 24 1400q summaries were recomputed (SLURM job 62425448,
COMPLETED 00:43:43). Below is the **old (buggy) → new (fixed)** diff at **round 0** (where the
distractors act), across all four models. Every shift is in the direction the audit predicted.

### Cohort size (union → intersection)
Uniform across all models/variants: `diverse_synth 1290 → 1199`, `equal_diverse_synth 1290 → 1170`.
~90–120 questions that were not seeded in *every* arm dropped out.

### `gold_match` @ round 0 (search variant)

| model | baseline (f=0) | f=0.7 |
|---|---|---|
| Qwen2.5-14B | 0.505 → **0.663** | 0.190 → 0.239 |
| Llama-3.1-8B | 0.448 → **0.641** | 0.088 → 0.301 |
| Mistral-7B | 0.204 → **0.609** | 0.040 → 0.265 |
| **DeepSeek-R1-7B** | **0.013 → 0.518** | **0.004 → 0.373** |

DeepSeek was ~0.004–0.013 at *every* arm, a pure exact-match artifact; it is now real (~0.32–0.52).
The entire DeepSeek row of the previously published figures was an artifact.

### `distractor_adoption` @ round 0 (search variant)

| model | f=0 | f=0.3 | f=0.5 | f=0.7 |
|---|---|---|---|---|
| Qwen | 0.004→**0.000** | 0.524→0.659 | 0.510→0.661 | 0.507→0.660 |
| Mistral | 0.003→**0.000** | 0.233→0.729 | 0.167→0.758 | 0.136→**0.768** |
| DeepSeek | 0.000→0.000 | 0.014→0.673 | 0.010→0.734 | 0.011→0.739 |

Baseline adoption is now exactly 0 (was a spurious 0.002–0.006 from the union bug). Mistral's old
adoption *decreased* with dose (0.233→0.167→0.136), the union+exact bug inverting it; it now correctly
*increases* (0.729→0.758→0.768).

### Direction flipped (the scientific change)
- **diverse_synth accuracy**: old looked flat/slightly-*decreasing* → new clearly *rises* with fraction
  (e.g. Llama 0.233/0.275/0.301; DeepSeek 0.318/0.348/0.373). More competing single-doc falsehoods →
  the model falls back to the 2-doc-supported gold. Consistent with §6.
- **Caveat (honest):** `equal_diverse_synth` accuracy is **model-dependent**, not cleanly monotone: only Qwen degrades with more reinforced topics (0.193→0.165→0.145→0.155); Llama/Mistral/DeepSeek
  accuracy *rises* with topics. Present it per-model, not as a universal clean dose-response.

Figures regenerated from the recomputed summaries; Bug 8 (`sharey` on `distinct_answers`) fixed in
`make_1400q_figures.py`. Artifact `0b92b522` updated with corrected figures + narrative.
