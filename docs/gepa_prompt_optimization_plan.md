# GEPA System Prompt Optimization

## Overview

This subsystem uses [GEPA](https://gepa-ai.github.io/gepa/) (reflective prompt evolution) to optimize the RAG generation **system prompt** so that it resists answer collapse while staying faithful to the retrieved context.

**Goal:** evolve a model-agnostic system prompt that simultaneously:
1. Minimizes answer collapse across repeated self-refinement rounds (**anti-collapse**)
2. Maximizes answer quality relative to the original context (**faithfulness + relevance**)

GEPA treats the current hand-written system prompt as a *seed* candidate and uses an LLM ("reflection model") to read structured failure reports and propose mutations, selecting candidates on a **Pareto frontier** across the two objectives.

> **Code is the source of truth.** This doc reflects `gepa_optimization/` as committed. The mermaid diagrams in [`gepa_flowchart.md`](gepa_flowchart.md) are the visual companion.

---

## What is being optimized

- **File:** `formatters.py`
- **Variable:** `GEPA_RAG_GENERATION_SYSTEM_PROMPT`  (NOT the baseline `_RAG_GENERATION_SYSTEM_PROMPT`)

The seed is a multi-step *decompose → read broadly → read specifically → answer* strategy prompt (adapted from the agentic-RAG system prompt, minus tool use). `apply_best_prompt.py` writes the optimized result back into `GEPA_RAG_GENERATION_SYSTEM_PROMPT`; the baseline one-liner `_RAG_GENERATION_SYSTEM_PROMPT` is left untouched so the two can be compared via the pipeline's `--use-gepa-prompt` flag.

```python
seed_candidate = {"system_prompt": GEPA_RAG_GENERATION_SYSTEM_PROMPT}
```

---

## Dataset preparation (`gepa_optimization/prepare_dataset.py`)

**Sources (100 questions total):**
- 50 questions from `datasets/umass_data.entity.chatgpt.400.jsonl` (web-scraped reference docs already attached).
- 50 questions from **HotpotQA**, whose context docs are **retrieved at prep time** by encoding the `mteb/hotpotqa` corpus with `intfloat/e5-small-v2` and building a **FAISS** index (full runs use `IndexIVFFlat`; `--smoke-test` uses a tiny `IndexFlatIP`). Top-`k` (default 10) passages are retrieved per question.

Combined, shuffled with `SEED=42`, split **60 / 20 / 20** → `gepa_optimization/data/{train,val,test}.jsonl`.

| Split | Size | Purpose |
|-------|------|---------|
| train | 60 | evaluated during each GEPA mutation round |
| val   | 20 | used by GEPA to rank candidates (Pareto) |
| test  | 20 | held out; for final unbiased reporting |

Each JSONL line is a `RAGDataInst`:
```python
@dataclass
class RAGDataInst:
    question: str
    docs: list[dict]   # each always has "text"; UMass docs add "url"; HotpotQA docs add "doc_id"="corpus_<id>", "title"
```
`title`/`doc_id` are metadata only. `references_to_documents()` generates its own `doc_id` and propagates `url` + `text` into the pipeline.

**Run it (GPU, one-time):**
```bash
sbatch scripts/gepa_prepare_dataset.sh
# or directly:
python gepa_optimization/prepare_dataset.py --cache-dir <hf_cache> --index-dir <faiss_dir>
python gepa_optimization/prepare_dataset.py --smoke-test    # tiny, no GPU index build
```
The FAISS index is saved to `--index-dir` (scratch) and reused on later runs.

---

## Scoring (`gepa_optimization/scoring.py`)

Thresholds: `ANTI_COLLAPSE_THRESHOLD = 0.30`, `QUALITY_THRESHOLD = 0.50`.

### Anti-collapse (higher = more diverse = less collapse)
`anti_collapse_score(question, answers, judge_model, …) → (score, unique_entity_count)`
1. Extract named entities from each round's answer via the judge LLM (temperature 0).
2. Cluster surface mentions into canonical forms via the judge LLM.
3. Build a binary entity-mention vector per answer.
4. `mean_entity_similarity` = mean pairwise cosine over those vectors.
5. `score = 1.0 − mean_entity_similarity`.

Falls back to `(1.0, 0)` if fewer than 2 non-empty answers or no mentions. Range `[0,1]`; near 1 = diverse entity sets across rounds, near 0 = collapsed.

### Quality (higher = better grounded)
`judge_quality_score(question, original_context, final_answer, …) → float`
One judge call comparing the **final-round answer** against the **original (uncontaminated) round-0 context**:
```
quality = 0.6 × faithfulness + 0.4 × relevance
```
Faithfulness is weighted higher because grounding is the primary concern. Falls back to `0.5` on parse error.

---

## Adapter (`gepa_optimization/rag_adapter.py`)

`RAGSystemPromptAdapter(GEPAAdapter)`, constructed with `task_model, doc_gen_model, judge_model, api_base, api_key, n_rounds=10, embed_model, chars_per_doc=800, logger`.

`evaluate(batch, candidate, capture_traces)` for each `RAGDataInst`:
1. Build `original_context` from `docs` (kept as ground truth for quality judging).
2. Run the three collapse simulations (`pipeline_simulator.py`), injecting `candidate["system_prompt"]` each round: `simulate_replace_all`, `simulate_replace_one`, `simulate_search` (all `n_rounds=10`).
3. Score each variant with `anti_collapse_score` + `judge_quality_score`.
4. Average across the three variants.
5. Return `EvaluationBatch(outputs, scores, trajectories, objective_scores)` where `scores = [avg_anti_collapse, …]` and `objective_scores = [{"anti_collapse":…, "quality":…}, …]` (multi-objective → Pareto).

`make_reflective_dataset(candidate, eval_batch, components_to_update)` builds, per question, a record:
- **`Inputs`**: `{"System Prompt": candidate["system_prompt"]}`, *only the system prompt* (the question, context, and per-round structure are deliberately withheld to prevent the reflection LM from reward-hacking the round structure).
- **`Generated Outputs`**: per variant `{"answer": <final-round answer>}`, final answer only, not the round sequence.
- **`Feedback`**: per variant `{anti_collapse, unique_entities, quality, diagnosis}`, where `diagnosis` concatenates any triggered flags:
  - anti_collapse < 0.30 → *"The answer lacks entity diversity … the prompt may cause the model to fixate on a narrow subset of entities …"*
  - quality < 0.50 → *"Final answer drifted from original context …"*
- plus `scores`: `{avg_anti_collapse, avg_quality, avg_unique_entities}`.

---

## Collapse simulators (`gepa_optimization/pipeline_simulator.py`)

Reuse the real `pipeline/` context-building code; the LLM is `ProprietaryLLM` (LiteLLM → keymaker, no vLLM server). All run 10 rounds.

| Variant | Context update rule |
|---|---|
| `simulate_replace_all` | `HybridContextConfig(num_synth_docs=10, num_db_docs=0)`; round 0 = original refs, then the entire context becomes the previous answer |
| `simulate_replace_one` | start with refs (capped at `MAX_CITATIONS_REPLACE`=10); one slot replaced per round (`slot = iteration % len(docs)`) |
| `simulate_search` | `ChunkedRetrievalStore` seeded with refs, local `make_embed_fn_local(EMBED_MODEL)`, retrieves `SEARCH_TOP_K` each round; new answer added to the store |

---

## Running the optimization (`gepa_optimization/run_optimization.py`)

`run_optimization.py` takes **no CLI args**. It is configured entirely by environment variables.

```python
result = gepa.optimize(
    seed_candidate={"system_prompt": GEPA_RAG_GENERATION_SYSTEM_PROMPT},
    trainset=trainset, valset=valset,      # 60 / 20
    adapter=adapter,
    reflection_lm=reflection_lm,            # callable → REFLECTION_MODEL, temp 1.0
    max_metric_calls=MAX_METRIC_CALLS,      # default 300
    candidate_selection_strategy="pareto",
    display_progress_bar=True,
    run_dir=RUN_DIR,
)
best_score = result.val_aggregate_scores[result.best_idx]   # avg anti_collapse of best candidate
```

| Env var | Default | Role |
|---|---|---|
| `API_KEY` | *(required)* | keymaker API key |
| `LITELLM_API_BASE` | `https://thekeymaker.umass.edu/` | proxy for all model calls |
| `TASK_MODEL` | `openai/claude-haiku-4-5` | RAG generation under the candidate prompt |
| `DOC_GEN_MODEL` | `openai/gemma-3-12b-it` | generates the AI "contamination" documents |
| `JUDGE_MODEL` | `openai/gpt4o` | entity extraction, clustering, quality judging |
| `REFLECTION_MODEL` | `openai/claude-opus-4-1` | reads failure reports, proposes new prompts |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | local SentenceTransformer for `simulate_search` |
| `MAX_METRIC_CALLS` | `300` | GEPA evaluation budget |
| `GEPA_RUN_DIR` | `gepa_runs/rag_system_prompt_<timestamp>` | run output (logs, candidates, state, resumable) |

**Run it (CPU, all generation is via the API):**
```bash
export API_KEY="your-keymaker-key"
sbatch --export=ALL scripts/gepa_optimization.sh
```

---

## Inspecting & applying the result

```bash
# Rank candidates from a run's logs by mean avg_anti_collapse then avg_quality
python gepa_optimization/parse_results.py --run-dir gepa_runs/rag_system_prompt_<id> [--min-evals N]

# Write the chosen prompt back into formatters.py (GEPA_RAG_GENERATION_SYSTEM_PROMPT)
python gepa_optimization/apply_best_prompt.py --prompt '<optimized prompt>'   # or --stdin
```

Then verify on held-out data by running the **pipeline with the optimized prompt** and the standard metric scripts:
```bash
python pipeline.py … --use-gepa-prompt          # uses GEPA_RAG_GENERATION_SYSTEM_PROMPT
python evaluation.py <experiment.json> <eval.json>
python entity_extraction.py --experiment-files <experiment.json> …
```
Compare collapse/quality metrics against a baseline run (same flags, without `--use-gepa-prompt`).

---

## File layout

```
gepa_optimization/
├── prepare_dataset.py     # UMass + HotpotQA (E5+FAISS retrieval) → data/{train,val,test}.jsonl
├── rag_adapter.py         # RAGSystemPromptAdapter (GEPAAdapter)
├── scoring.py             # anti_collapse_score(), judge_quality_score()
├── pipeline_simulator.py  # simulate_replace_all / replace_one / search
├── run_optimization.py    # entry point (env-var driven)
├── parse_results.py       # rank candidates from a run's logs/
├── apply_best_prompt.py   # write best prompt into formatters.py
├── gepa_logger.py         # captures LLM calls + structured eval events
├── smoke_test_gepa.py     # tiny end-to-end smoke test
└── data/                  # train.jsonl / val.jsonl / test.jsonl

gepa_runs/rag_system_prompt_<id>/
├── logs/{aggregates,evaluations,prompts,run_meta}.jsonl
├── candidates.json, candidate_tree.html, gepa_state.bin   # resumable checkpoint
└── run_log.{json,txt}
```

---

## Results / caveats

- **Outcome so far:** the optimization runs to date did **not** beat the seed. The decompose-strategy seed prompt was already near the Pareto frontier on (anti-collapse, quality). Frame any reported numbers as "no improvement over an already-strong seed," not as a win.
- **Known code issues (not doc issues, flagged for fixing):**
  1. `scripts/gepa_optimization.sh` contains `export API_KEY=""` just before the key check, which blanks a key passed via `--export=ALL`. Set the key *after* that line or remove it.
  2. In `rag_adapter.py`, `evaluate()` runs the candidate prompt through `self._doc_gen_llm`; the `self._task_llm` built from `TASK_MODEL` is currently unused, so generation is effectively done by `DOC_GEN_MODEL`.
