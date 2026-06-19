# Handoff — Initial-doc "distractor" feature for the HotpotQA misinfo experiment

You are picking up a feature mid-design. This doc gives you everything: the repo, the experiment,
what's built, the conceptual dilemma, the decisions already made, the one decision still open, the
exact code hook points, and the constraints. Read it top to bottom, then ask the user the one open
question (or proceed with the recommended default) and implement.

---

## 0. TL;DR of the task

Add the ability to turn **a fraction of the round-0 *initial retrieved* documents into wrong-answer
"distractor" docs**, leaving the rest correct — so the starting answer distribution is wider and
partly-wrong, instead of a narrow spike on the gold answer. This is **separate from** the existing
`--inject-round` machinery (which corrupts the *generated* documents later in the loop). The goal is
to be able to actually observe accuracy *shift away from gold* across rounds, which the current
setup can't show (see §3).

- **Method already chosen by the user:** LLM rewrite (freeform-style) — have the doc-gen model
  rewrite a retrieved doc so it asserts the wrong answer (more naturalistic than editing the gold doc).
- **Fraction:** make it a flag (e.g. `--distractor-fraction`, default `0.5`), sweepable.
- **STILL OPEN (ask the user):** should the rewritten distractors all assert the **same** wrong
  entity (coordinated → clean bimodal start, exact ASR) or a **different** wrong entity each
  (diverse → max initial spread, messy/no single ASR target)? Recommended: **coordinated**. Also
  ask whether they want a free **`untargeted` noise arm** built from HotpotQA's native distractors
  (see §6).

---

## 1. Repo + branch state

- Repo: `RAG-Collapsement-on-Self-Refined-Generation` (research code on inference-time "RAG collapse").
- Read `CLAUDE.md` (repo guide) and `docs/misinfo_error_compounding.md` (the experiment) first.
- **Branch: `misinfo-error-compounding`** (the work lives here; draft PR **#32** to `main`).
- Latest commit on `origin`: `e9c9740` ("Add interactive HTML walkthrough…"). Recent relevant commits:
  per-model launchers (`82a905e`), param alignment (`6c06ec3`), parallel launcher (`b7d8444`),
  all-in-one smoke (`99143ba`).
- There is an interactive explainer at `docs/distractor_experiment.html` (open in a browser) that
  visualizes the whole experiment — useful for getting oriented fast.

---

## 2. What the misinfo experiment already does (so you extend, not duplicate)

The existing experiment lets the **document synthesizer** inject a falsehood into the doc it writes
each round, then tracks whether the seeded error propagates/amplifies/recovers once re-retrieved.
All default-off; `faithful` mode == the unchanged baseline.

- **3 synthesis modes** (`--doc-synthesis-mode {faithful,counterfactual,freeform}`): faithful
  (control), counterfactual (entity substitution, exact ground truth), freeform (LLM invents a false
  claim, a judge discovers it).
- **3 targets** (`--target-mode {final_answer,intermediate_hop,untargeted}`).
- **Other flags:** `--inject-round` (default 1), `--inject-every-round`, `--seed`, `--gt-file`,
  `--native-hotpot-file`.
- Per-question `injection` record written to the experiment JSON (schema-additive); injection-aware
  metrics in `hotpot_evaluation.py` (ASR = `injected_match_rate`, `gold_match_rate`,
  `retrieval_condition`, `doc_propagation`, `implied_match_rate`, aggregates incl. `asr_by_iteration`,
  `adoption_rate`, `recovery_rate`, …). No-op without injection records.
- **Validated:** smoke test passed on Unity (all arms run, parity holds, ASR curve non-degenerate).
  See `scripts/hotpot-misinfo/RUNBOOK.md`.

**The key reusable machinery is `pipeline/misinfo.py`** — it has exactly the helpers you need:
`substitute_in_text`, `choose_substitute`, `validate_substitute`, `realized_status`,
`contains_entity`, `normalize`, `load_ground_truth`, `load_native_records`, `bridge_entity_for`,
the `InjectionRecord` dataclass, and the `MisinfoController` class (modes `MODE_*`, targets
`TARGET_*`). Prompts (including the freeform create-doc + judge prompts) live in `formatters.py`
(`get_create_document_conversation*`, `get_substitute_proposal_conversation`,
`get_claim_discovery_conversation`, `get_implied_answer_conversation`).

---

## 3. The conceptual dilemma (why this feature exists)

From a design discussion with the user/collaborator. Frame "collapse" as two things:
**(a) concentration** (distribution narrows, diversity drops) and **(b) shift** (the mode moves).

- **Entity prompts** (the `umass_data.entity.*` dataset, subjective questions): wide support → collapse
  shows as *concentration*. That's the diversity story (already studied).
- **Factual / HotpotQA**: support is near-degenerate (one right answer), so concentration has nowhere
  to go — the only interesting axis is the *shift* off gold.
- **Why the project's factual runs were flat:** the initial context was seeded with **correct-answer
  docs**, so round 0 is already a narrow spike on gold. Iterating a loop whose fixed point *is* the
  truth is a no-op — you can't observe drift if you start on the attractor. To study the shift you
  must move the starting distribution off-gold **or** perturb the loop.

Two complementary ways to do that:
1. **Distractor docs** (THIS feature): put wrong info in the *initial retrieved* context → wider,
   partly-wrong round-0 distribution → can shift away from gold over rounds.
2. **Generated-stream injection** (already built, `--inject-round`): perturb the loop endogenously →
   tests whether the loop *amplifies* an error once it's in the self-generated doc stream.

**Why both, not just one:** distractor docs alone test *static RAG robustness* (can bad retrieval fool
the model at round 0 — well-studied, e.g. PoisonedRAG); injection tests *loop amplification* (the
project's actual novel thesis). The strongest experiment combines them, and the decisive contrast is:
introduce a distractor that fools the model at round 0, then **remove it**, and see whether the
adopted error *persists/amplifies via the model's own generated docs* (real compounding) or *decays*
(loop not self-reinforcing). Keep that experiment in mind — this feature is the first half of it.

Caveat the user is aware of: hand-forced errors can shade into "does the model follow authoritative
context" (context-faithfulness) rather than emergent compounding. The `untargeted` noise arm and the
matched faithful control bound that. Lead with the most naturalistic arm in any writeup.

---

## 4. Does HotpotQA already provide distractors? (researched — important)

**Yes, but not the kind needed here.** HotpotQA's official *distractor setting* gives 10 context
paragraphs per question = **2 gold supporting + 8 bigram-TF-IDF distractors**. The distractors are
*challenging noise that lacks the answer* — they do **not** assert a *wrong* answer, so they widen the
start as **noise/off-topic**, not as a competing **wrong attractor** pointing at a specific incorrect
entity.

Verified on Unity: our file `hotpot_dev_fullwiki_v1.json` (7,405 records) has keys
`_id, answer, question, supporting_facts, context, type, level`; `context` is a list of **10
paragraphs** (`[title, [sentences…]]`), and the gold `supporting_facts` titles are mostly *absent*
from those 10 — i.e. they're TF-IDF-retrieved hard negatives (fullwiki), not guaranteed gold.

**Two wrinkles:**
1. Our pipeline does **not** use the native `context` field for retrieval at all — it retrieves from a
   **FAISS Wikipedia index** (`build_hotpotqa_index.py`) via mteb/hotpotqa queries, and reads the
   native JSON only for gold answers (`load_ground_truth`) + supporting_facts (hop target).
2. HotpotQA's distractors have no answer-level ground truth ("what wrong answer does this imply"), so
   no clean ASR off them.

**Conclusion:** for wrong-answer-asserting distractors (what the user wants) we **create our own** via
LLM rewrite. But HotpotQA's native distractors are useful as: (a) a **free `untargeted`/noise control
arm** (real related docs, no false claim), and (b) optionally the **raw material** to rewrite — taking
a real distractor/retrieved paragraph and rewriting it to assert the wrong answer is more naturalistic
than editing the gold doc.

Sources: HotpotQA paper (Yang et al. 2018, EMNLP); alphaXiv/EmergentMind benchmark overviews.

---

## 5. Exact code hook points (verified line numbers, branch `misinfo-error-compounding`)

`hotpot_pipeline.py`:
- **Initial retrieval / round-0 docs** (where you inject initial distractors): ~L295 "Running initial
  retrieval…", L309 `s.current_docs = list(s.initial_corpus_docs)`, L312
  `s.current_docs = _select(s.initial_corpus_docs, args.num_db_docs, args.db_doc_selection)`. The
  per-question `PipelineState` carries `initial_corpus_docs` and `current_docs`. **This is the place
  to corrupt a fraction of the round-0 docs before the first answer is generated.**
- **Round loop:** `for it in range(num_iterations)` ~L406; answer built from `s.current_docs` (L414);
  per-round `"documents": s.current_docs` recorded (L439).
- **Doc-synthesis hook (answer→doc):** L458 `ans_for_docs = [random.choice(answers)]`; L461
  `controller.make_doc_conversation(...)` vs L463 `get_create_document_conversation(...)`.
- **Controller lifecycle:** built L367, `controller.prepare(states)` L379, `should_inject(it)` L297,
  `make_doc_conversation` L461, `record_injection(...)` L506, `attach(states)` L509. Flags parsed
  ~L199–L236; `inject_enabled = args.doc_synthesis_mode != MODE_FAITHFUL` L235.
- **AI-doc tagging convention:** generated docs get `doc_id="gen_{iter}_{i}"`, `url="model_generated"`;
  corpus docs `doc_id="corpus_{did}"`. Every injection metric keys off these tags. Your initial
  distractors will be *corpus* docs that you mutate at round 0 — decide how to tag/track them (e.g. a
  `distractor=True` flag on the doc dict + in the injection record) so the eval can find them.

`pipeline/misinfo.py` public surface (reuse these): `normalize` L69, `contains_entity` L82,
`substitute_in_text` L93, `choose_substitute` L180, `validate_substitute` L157, `realized_status`
L194, `load_ground_truth` L132, `load_native_records` L146, `bridge_entity_for` L208,
`InjectionRecord` L231, `MisinfoController` L254 (modes/targets constants L54–L62).

---

## 6. Suggested implementation (starting point — confirm the open decision first)

1. **New flags** in `hotpot_pipeline.py` (all default-off so faithful stays byte-identical):
   `--distractor-fraction FLOAT` (default 0.0 = off; 0.5 typical), `--distractor-mode
   {rewrite,substitution,native_noise}` (user chose **rewrite**; keep the others as options),
   reuse `--seed` + `--gt-file`. Hard-error if `--distractor-fraction>0` and no `--gt-file`.
2. **At round 0** (hook L309/L312), for each question: pick which of the `current_docs` to corrupt —
   prefer docs that contain the gold entity (so the change is meaningful); corrupt
   `round(fraction * k)` of them via the chosen method, leave the rest intact. Use the per-question
   seeded RNG (`MisinfoController._rng`, L294) so arms are reproducible and the faithful/treatment
   selection is matched.
   - **rewrite**: build a doc-gen conversation that rewrites the doc to assert the wrong answer (new
     prompt in `formatters.py`, sibling to the freeform create-doc prompt); batch through the
     doc-LLM; verify with `realized_status`.
   - **substitution**: reuse `substitute_in_text` + `choose_substitute` directly (cheaper, exact GT).
   - **native_noise**: pull paragraphs from the native `context` field (load via `load_native_records`)
     — the `untargeted` control; no false claim, just real non-gold docs.
3. **Coordination (OPEN DECISION):** if coordinated, choose one wrong entity per question (via
   `choose_substitute`) and make all that question's distractors assert it; if diverse, choose a
   different one per doc. Coordinated keeps ASR exact. **Default to coordinated unless the user says
   otherwise.**
4. **Record** the initial distractors in the per-question `injection` record (extend it — it's
   schema-additive): e.g. `initial_distractor: {fraction, n_corrupted, n_docs, injected_entity(ies),
   corrupted_doc_ids, mode}`. Tag the doc dicts so the eval can detect them per round.
5. **Eval** (`hotpot_evaluation.py`): the existing injection metrics mostly transfer (ASR / gold /
   retrieval_condition already key off the injected entity). Make sure `retrieval_condition` /
   propagation also recognize the initial-distractor doc ids. Add a round-0 ASR/gold point so you can
   see the *shift from a wider start*.
6. **Tests:** add cases to `tests/test_misinfo.py` (pure-Python, no GPU) — fraction math, which-docs
   selection, coordinated vs diverse entity assignment, record round-trip. These run locally; see §7.
7. **Smoke:** extend `scripts/hotpot-misinfo/smoke_test.sh` (or add a flag passthrough in
   `run_variant_client.sh` + the launchers) so the new arm gets exercised, then run the all-in-one
   smoke per the RUNBOOK.

Keep `faithful` + `--distractor-fraction 0` byte-identical to the current pipeline.

---

## 7. Constraints & gotchas (will bite you if ignored)

- **Unity GitHub auth:** the Unity clone's `origin` is HTTPS with **no stored credential** — an agent
  session **cannot `git pull`/`push`** there (needs the user's PAT). Workflow this session used:
  commit + push from the **laptop** clone (`C:\Users\riddh\RAG-Collapsement-on-Self-Refined-Generation`),
  then the **user** pulls on Unity. Don't try to push from Unity.
- **Run env:** Unity conda env **`ragenv`** (has vllm/faiss/torch). Laptop has **`rag-collapse`** but
  **no GPU / no vLLM server** — only good for `py_compile`, `pytest tests/test_misinfo.py`, and
  arg-parse/wiring checks (the pure-Python misinfo tests pass there).
- **2-day SLURM job cap.** Launchers already handle it: servers request 48h but a reaper `scancel`s
  them when clients finish; clients request 47h so servers always outlive them.
- **vLLM 0.20.0, SLURM 25.11.** DeepSeek-R1 needs `--reasoning-parser deepseek_r1` (keeps `<think>`
  out of `message.content`) + larger answer `--max-tokens`; Mistral needs `--tokenizer-mode mistral`.
  These are set by the per-model launchers. Non-agentic variants need NO tool-call flags.
- **Canonical run params** (match these for comparability — already aligned in the misinfo scripts):
  `--num-runs 10`, `--chars-per-doc 500`, answer `--max-num-seqs` 64 for 14B / 128 for 7-8B,
  `DEFAULT_ROUNDS` search 30 / replace_one 20 / hybrid 10.
- **Never commit API keys.** `CACHE_DIR` in the SLURM scripts must stay hardcoded to
  `/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache` (do NOT switch to `$USER`).
- **Empty-entity convention here:** entity similarity for an empty-entity answer is scored 0.0 (a
  divergence from the sister `collapse-randomness-research` repo, which uses 1.0).

---

## 8. How to run the existing experiment (for reference / to test your changes)

All under `scripts/hotpot-misinfo/` (see `RUNBOOK.md` for the full walkthrough):
- **Quick smoke (1 GPU, ~8 min):** `sbatch --export=ALL,MAX_Q=20 scripts/hotpot-misinfo/smoke_all_in_one.sh`
- **Per-model launchers (login node; 2 GPUs each, all 3 variants × 3 arms, shared servers + reaper):**
  ```
  export GT_FILE=/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpot_dev_fullwiki_v1.json
  bash scripts/hotpot-misinfo/launch_qwen14b.sh   # also _mistral7b / _llama8b / _deepseek7b
  ```
- Outputs land per-model under
  `/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hotpotqa_distractor_experiment/<served-name>/`.
- Data artifacts on Unity: FAISS index `…/hotpotqa_index/{ivf.index,docid_map.json}`; gold/native
  `…/hotpot_dev_fullwiki_v1.json`; queries `…/hf_cache/queries/{test,train,val}`.

---

## 9. First thing to do in the fresh session

1. Read `docs/misinfo_error_compounding.md` and skim `pipeline/misinfo.py` + the §5 hook points.
2. Ask the user the **one open decision** (coordinated vs diverse wrong entity; and whether to add the
   native-distractor `untargeted` noise arm). Default to **coordinated**, method **rewrite**, fraction
   a sweepable flag (default 0.5).
3. Implement per §6, keep faithful/`--distractor-fraction 0` byte-identical, add pure-Python tests,
   then commit on `misinfo-error-compounding` from the laptop clone and let the user pull on Unity.
4. (Optional, high value) wire up the "introduce distractor → remove it → does the error persist?"
   contrast from §3 — it directly answers whether the loop truly *compounds* vs merely *follows
   bad context*.
