# Experimental Pipeline: `RAG-Collapsement` vs. `collapse-randomness-research`

A detailed compare-and-contrast of the two RAG-collapse experiment pipelines, focused on the **baseline experiments** (replace-all / replace-one / search) and the **metrics**.

| | This repo | Reference repo |
|---|---|---|
| Name | `RAG-Collapse` | `collapse-randomness-research` (graphite-growth) |
| Baseline driver | `pipeline.py::run_pipeline` | `model_collapse/core/model_collapse.py::main` → `model_collapse/core/simulation.py::run_simulation` |
| Metrics | split: `evaluation.py` (text metrics) + `entity_extraction.py` (entity metrics), run as **separate programs** | `randomness/core/metrics.py::simulation_metrics` (+ entity fns), computed **inline** during the sim loop |
| Generation models | **open / local** (Mistral-7B, Qwen2.5-14B, Llama-3.1-8B, DeepSeek-R1-7B) via vLLM | **frontier APIs** (`gpt-5.2-chat-latest`, Claude, Gemini) via OpenAI Responses API |
| Real corpus | pre-built dataset with references attached | **live-scraped** citation URLs (Zyte + trafilatura + LLM filter) |

> Orientation on the reference repo: it has two subsystems. **`model_collapse/`** is the iterative collapse pipeline compared here. **`randomness/`** is a *separate* single-shot non-determinism/variance study , but its `randomness/core/{data,generate,metrics}.py` are the **shared library** the collapse pipeline imports (so `randomness/core/metrics.py` is the canonical metrics module for both).

---

## 1. High-level architecture

**This repo** separates concerns into three programs:
1. `pipeline.py` runs the generation loop → writes `experiment_outputs/*.json` (raw answers + docs per round).
2. `evaluation.py` reads that → writes `evaluation_outputs/*_eval.json` (text-similarity metrics).
3. `entity_extraction.py` reads that → writes `entity_extraction_output/*_entity_results.json` (entity metrics / collapse).

**Reference repo** is a single pass: `run_simulation` runs the loop **and computes all metrics inline** (`simulation_metrics`, `get_entities_by_round`), emitting one combined JSONL record per question with both raw generations and the per-round metric arrays. Entity re-tagging / ranking similarity are the only post-hoc steps (`fix_entity_counts.py`, `add_ranking_similarity.py`).

Consequence: this repo's metrics are recomputable from stored outputs without re-running generation (cheaper iteration, and why the consolidated dataset keeps the three trees separate). The reference repo couples generation + measurement, so changing a metric means re-running (expensive, since generation is frontier-API).

**Loop ordering.** This repo is **iteration-major across all questions** (batch every question's `num_runs` answers for round *r*, then advance), built for vLLM batch throughput. The reference is **question-major** (`main` loops questions; `run_simulation` runs all rounds for one question) with per-round `multiprocessing.Pool` over the 10 generations, built for parallel API calls.

---

## 2. Corpus construction (where the documents come from)

| | This repo | Reference repo |
|---|---|---|
| Real docs | Supplied in the dataset JSONL (`{"question", "references":[{"url","text"}]}`), `data_loader.py::load_dataset` | **Scraped** from the question's citation URLs: Zyte API render (`scrape.py::zyte_parallel_download`) → trafilatura extract → **LLM main-content filter** (`filter_main_content`, `gpt-5.2`), cached to a JSONL page cache |
| Min citations | skip if `< 5` refs (`MIN_CITATIONS`) | skip if `< 5` citations, and skip if `< 5` survive scraping |
| Doc length control | `--chars-per-doc` (e.g. 400) truncation when building context | sort pages by relevant-text length, truncate to `citation_limit` (10) |
| Synthetic "AI" doc | `formatters.py::get_create_document_conversation` , "professional content writer" rewrites (question, answer) into a web doc | `simulation.py::get_replacement_with_generation` , rewrites a generation into an `Article` via `convert_to_article` (`translate`/`expand`/`expand_long`) |
| AI tag | `doc_id="gen_{iter}_{i}"`, `url="model_generated"` (`feedback_loop.py`) | `url=f"generation_{n}"` (`simulation.py`) |

Both **tag AI documents with a synthetic id/URL**, and every AI-contamination metric keys off that tag (not content classification). The big difference is the **source of truth for real docs**: the reference repo does its own live scraping + LLM cleaning (heavy, costly, frontier-model-dependent); this repo trusts a curated dataset.

---

## 3. Baseline experiment variants

Both implement the same three corpus-evolution regimes; they differ only in the *update rule* applied to the document pool each round.

| Variant | This repo (`context_builder.py` / `feedback_loop.py`) | Reference (`simulation.py`) | Same idea? |
|---|---|---|---|
| **replace_all** | hybrid config `num_synth_docs=10, num_db_docs=0`: entire context replaced by 10 fresh AI docs each round; refs gone after round 0 | `replace_all=true` → `round_increment=len(pages)`; every page overwritten in place each round | ✅ identical intent |
| **replace_one** | one slot per round, round-robin `iteration % len(docs)`, rest persists; refs (≤10) gradually displaced | `replace_all=false, search=false` → `round_increment=1.0`; `pages[index]=replacement`, index cycles | ✅ identical |
| **search** | `ChunkedRetrievalStore` (local SentenceTransformer `all-MiniLM-L6-v2`, char chunks 500/overlap 50, cosine top-k=10); AI docs **appended** to a growing pool, only surface if retrieved | OpenAI **vector store** over pages; AI docs `pages.append` + `upload_files_to_vector_store`; retrieve chunks per round | ✅ same design, different retrieval backend |

Shared design choices in both:
- replace_all = full overwrite, replace_one = single-slot overwrite (fixed corpus size), search = **additive growing pool** + retrieval (the only regime where AI docs *compete* rather than being force-fed.
- Per-round, **one** randomly chosen answer becomes the new AI doc for replace_one/search (this repo: `random.choice`; reference: cycling index), whereas **all** answers become docs in replace_all.
- A `stop_if_converged` early-stop exists in both, **off by default**.

Key backend divergence in `search`: this repo retrieves with an **in-memory normalized-cosine** store over local embeddings; the reference delegates chunking + retrieval to **OpenAI's vector store**. The reference also separates `shuffled_pages` (full pool) from `shuffled_inputs` (retrieved subset) so it can measure retrieval dynamics (see §6); this repo records only the retrieved `documents` per round.

Axes only the reference has: **entity vs editorial** dataset (`use_entities` toggles the whole entity-metric suite), and **article-prompt style** (`translate`/`expand`/`expand_long`). This repo fixes the entity dataset (umass entity-comparison 400) and a single create-document prompt. This repo's extra axis is **agentic_rag** (model issues its own retrieval tool calls) , no analog in the reference.

---

## 4. The round / iteration loop

| | This repo | Reference |
|---|---|---|
| Rounds | replace_all **10**, replace_one **20**, search **30** (`config.py::get_rounds_for_variant`) | replace_all **10**, replace_one **20**, search **20** (config JSONs) |
| Generations/round | **10** (`--num-runs`, `RUNS_PER_ROUND`) | **10** (`num_generations`) |
| Diversity source | sampling **temperature 0.7** (top_p 0.9), same prompt across 10 runs | model default sampling + **shuffling** pages each round + fresh `prompt_cache_key`; temperature not explicitly set |
| Parallelism | one big `llm.inference_batch` across all questions×runs (vLLM) | `multiprocessing.Pool` over the 10 generations per question; one warm-up "first" call |
| AI-doc accumulation | replace_all none (full refresh), replace_one 1 slot/round, search +1 to pool/round | same |

Both stop early only if `stop_if_converged` and 4 consecutive fully-collapsed rounds.

---

## 5. Models & generation backend

- **This repo** targets **open-weight models served by vLLM** . `server` mode (`ServerLLM`, OpenAI-compatible HTTP) is the production path; `local` (in-process vLLM) and `api` (LiteLLM via `thekeymaker.umass.edu`) also exist. Default experiment model **Mistral-7B-Instruct-v0.3**; the project sweeps Qwen2.5-14B, Llama-3.1-8B, DeepSeek-R1-7B. Generation `temperature=0.7, max_tokens=512, top_p=0.9`. Embeddings: local SentenceTransformer `all-MiniLM-L6-v2`. A separate doc-generation model can be wired (`--doc-model-mode`, second vLLM server).
- **Reference repo** targets **frontier APIs**. `generate.py` dispatches by model-name substring to OpenAI **Responses API** (`gpt-5.2-chat-latest`), Anthropic, or Gemini, all with structured output + prompt caching. Metrics/judge/scrape-filter model = **`gpt-5.2`**; embeddings = **`text-embedding-3-small`**. Global **cost tracking** (`increment_cost`) across generation, scraping, entity extraction, dedup, judging, embeddings.

This is the deepest conceptual difference: **the reference studies collapse in frontier closed models; this repo studies it in small open models** (and adds an agentic variant). It also drives the metric-model gap below (Qwen-7B judge vs gpt-5.2 judge).

---

## 6. Metrics : the core comparison

Both measure collapse as **convergence of the 10 per-round answers toward each other** plus **AI self-reinforcement**. But the metric sets only partially overlap.

### 6a. Inter-answer similarity

| Metric | This repo | Reference |
|---|---|---|
| Whole-answer embedding cosine | ✅ `avg/max/min/std_pairwise_similarity` (`calculate_pairwise_similarities`) | ✅ `mean similarity` (embeddings of generations) |
| Sentence-level embedding (TES) | ✅ `avg/std_pairwise_tes` , truncate to min sentence count, per-sentence aligned cosine | ❌ none |
| ROUGE-1/2/L (F-measure) | ✅ `avg/std_pairwise_rouge1/2/L` | ❌ none |
| Generations-to-input similarity | ❌ | ✅ `mean similarity generations to input` (round-0 inputs × current gens) |

This repo adds **lexical (ROUGE) and sentence-aligned (TES)** similarity the reference lacks. The reference instead tracks drift **away from the original inputs**.

### 6b. Lexical diversity & length

| Metric | This repo | Reference |
|---|---|---|
| Unique words (union across answers) | ✅ `unique_words` | ✅ `unique words per round` |
| Unique words per generation | ❌ | ✅ |
| Answer length mean / stddev | ❌ | ✅ `length words`, `stddev length words` |

### 6c. Semantic-duplicate judge (same answer)

Both: sample **10** answer pairs, ask an LLM **"are these paraphrases?"**, score = fraction yes.
- This repo: `calculate_same_answer_percentage`, **seeded** (seed 42), judge = **Qwen2.5-7B** (small, local), `temperature 0`, returns 0–100.
- Reference: `get_same_answer_score`, judge = **gpt-5.2** (frontier), returns 0–1; its accuracy is human-validated in `evaluate_same_answer.py`.

Same algorithm, **very different judge quality** . The reference's frontier judge is more reliable; this repo's choice is constrained by cost/local-only.

### 6d. Entity metrics (the collapse signal)

This repo computes these in the **separate** `entity_extraction.py`; the reference computes them **inline**. (A full function-level diff is in `docs/entity_extraction_comparison.md`.) Summary:

| Metric | This repo | Reference |
|---|---|---|
| Unique entities / round | ✅ count of canonical entities | ✅ `entities` = `len(entity_counts)` |
| Entity similarity (binary cosine) | ✅ mean/min/max/std; **empty answer → 0.0** | ✅ `mean entity similarity`; **empty answer → 1.0** |
| Entity entropy | ❌ | ✅ `entity entropy` (Shannon over counts) |
| Ranking similarity (Kendall-τ) | ❌ | ✅ `ranking similarity` (positional order, NaN → 1.0) |
| AI entity influence | ❌ | ✅ `input_to_generation_entity_similarity` (AI-input share of generation entities) |
| Extraction model | configurable (gpt-4o / local), question-conditioned, JSON-parsed | `gpt-5.2`, question-agnostic, structured output |

⚠️ **Opposite empty-vector convention** (carried over from the metrics modules): for an answer with no entities, the reference scores the pair **1.0 (collapsed)** while this repo scores **0.0 (diverse)** . These push the collapse curve in opposite directions for degenerate output. Same caveat applies to the reference's "no-citation → 1.0" and "Kendall-τ NaN → 1.0".

### 6e. AI self-reinforcement / contamination

| Metric | This repo | Reference |
|---|---|---|
| AI share of context | ✅ `ai_reference_percentage` = fraction of round's docs with `gen_` id (0–1) | ⟶ embedded in `expected` (AI fraction of inputs) |
| AI citation rate | ✅ `ai_citation_percentage` (needs citations enabled) | ✅ `ai citations` + **`ai citations over expected`** (= rate/expected − 1) and **by length** |
| AI retrieval rate (search) | ❌ | ✅ `ai retrieval rate`, `expected ai retrieval rate`, `over expected` |
| AI context length | ❌ | ✅ `mean context length ai` vs `orig` |

The reference has a **richer self-reinforcement suite** . It normalizes AI-citation and AI-retrieval rates against their *expected* prevalence (does the model cite/retrieve AI docs **more than** their share?), which is the sharper collapse signal. This repo reports the raw AI fraction and a raw AI-citation %.

### 6f. AI-content detection method

Both detect "AI" by **document metadata tag** (`gen_`/`model_generated` here; `generation_` prefix there) , not content analysis. This repo additionally ships a neural detector (`desklib/ai-text-detector`, `ai_detector.py`) but uses it **only at generation time** for a mitigation prompt, **not** in evaluation.

---

## 7. Output schemas

**This repo** : nested JSON, generation and eval separated:
```
experiment: questions[ {question_id, question_text, iterations[ {iteration_number,
            documents[{doc_id,iteration,url,text}], runs[{run_id,answer,citations,...}] } ] } ]
eval:       questions[ {question_id, iterations[ {iteration_number, metrics{...}} ] } ],
            aggregate_statistics{avg_collapse_rate: 1}   # ⚠ hardcoded placeholder
```
Cross-round aggregation is **not** done in `evaluation.py` (placeholder = 1); the notebook does it from per-iteration metrics.

**Reference repo** : one JSONL line per question, generation + metrics combined:
```
{question, generations[round][gen], replacements, entity_counts, entity_map,
 config, metrics{name:[per-round]}, raw_entities, input_generation_sims,
 raw_input_entities, git_commit, original_pages, shuffled_pages, retrievals,
 converged_in_round}
```
It stores **per-round metric arrays**, the full corpus evolution (`replacements`, `shuffled_pages`, `retrievals`), and the **git commit** for provenance . Richer self-contained records, at the cost of coupling.

---

## 8. Summary: what's the same, what diverges

**Same (the experimental design is genuinely parallel):**
- Three collapse regimes : replace_all (full overwrite), replace_one (single-slot), search (growing retrievable pool) , with identical update semantics.
- 10 generations/round; replace_all 10 rounds, replace_one 20 rounds.
- AI docs created by rewriting answers into web-doc/article form, tagged with synthetic ids, reinjected; AI detection via those tags.
- Collapse measured by answer-similarity convergence + a 10-pair LLM paraphrase judge + entity-set similarity + AI self-reference.

**Diverges (matters for results / comparability):**
1. **Models**: open/local (Mistral/Qwen/Llama/DeepSeek via vLLM) vs frontier APIs (gpt-5.2/Claude/Gemini). Different collapse dynamics entirely.
2. **Corpus source**: curated dataset references vs live Zyte+trafilatura+LLM scraping of citation URLs.
3. **Search backend**: local SentenceTransformer cosine store vs OpenAI vector store.
4. **Metric sets**: this repo adds ROUGE + TES (lexical/sentence); the reference adds entity entropy, Kendall-τ ranking, AI entity influence, AI-citation/retrieval-over-expected, context-length . A richer self-reinforcement and ordering suite.
5. **Judge / entity model**: Qwen2.5-7B (local) vs gpt-5.2 (frontier) . Large reliability gap.
6. **Empty/degenerate conventions**: reference biases empty/NaN → 1.0 (collapsed) for entity-similarity, AI-citations, Kendall-τ; this repo's entity similarity uses 0.0 (diverse) and AI-reference is a plain metadata fraction. **These can flip the collapse signal** and should be reconciled before cross-repo comparison.
7. **Aggregation**: reference aggregates per-round inline + stores arrays; this repo leaves aggregation to the notebook and has a hardcoded `avg_collapse_rate` placeholder.
8. **Extra scope**: this repo adds an **agentic_rag** variant and a 30-round search; the reference adds **entity-vs-editorial** datasets, **article-prompt** styles, and rerank/oracle ablations.

**If aligning numbers across the two repos**, the highest-impact items are (5) the judge/entity model, (6) the empty-vector conventions, and the entity-extraction differences detailed in `docs/entity_extraction_comparison.md` (question-conditioned vs question-agnostic extraction).
