# GEPA System Prompt Optimization Plan

## Overview

This plan describes how to use [GEPA](https://gepa-ai.github.io/gepa/) (Generative Evolutionary Prompt Adaptation) to automatically optimize `_RAG_GENERATION_SYSTEM_PROMPT` in `formatters.py`.

**Goal:** Evolve a model-agnostic system prompt that simultaneously:
1. Minimizes answer collapse across repeated runs (anti-collapse)
2. Maximizes answer quality relative to retrieved context (faithfulness + relevance)

GEPA treats the current hand-written system prompt as a seed and uses LLM-guided reflection to propose and evaluate mutations, selecting candidates on a Pareto frontier across both objectives.

---

## What Is Being Optimized

**File:** `formatters.py`
**Variable:** `_RAG_GENERATION_SYSTEM_PROMPT`

Current value:
```
You are a helpful AI assistant that answers questions using only the information provided
in the given context. You provide accurate, well-grounded responses based solely on the
retrieved documents.
```

The user prompt template (`_RAG_GENERATION_USER_PROMPT`) is **not** in scope — only the system prompt.

`seed_candidate` for GEPA:
```python
seed_candidate = {
    "system_prompt": _RAG_GENERATION_SYSTEM_PROMPT
}
```

---

## Architecture

```
datasets/umass_data.entity.chatgpt.400.jsonl  (50 questions)
hotpot_qa distractor train split              (50 questions)
        │
        ▼
  prepare_dataset.py
  shuffle seed=42, split 60/20/20
  → data/train.jsonl (60)
  → data/val.jsonl   (20)
  → data/test.jsonl  (20)
        │
        ▼
  RAGSystemPromptAdapter (rag_adapter.py)
  ┌──────────────────────────────────────────┐
  │  For each (question, docs):              │
  │    Simulate 3 collapse variants × 10 rds │  ← candidate system_prompt injected each round
  │    replace_all | replace_one | search    │
  │         │                                │
  │    ┌────┴─────┐                          │
  │    │ Score 1  │  Anti-collapse: 1 − mean_entity_similarity across rounds
  │    │ Score 2  │  Quality: 0.6×faithfulness + 0.4×relevance vs original context
  │    └──────────┘                          │
  └──────────────────────────────────────────┘
        │
        ▼
  gepa.optimize(
    seed_candidate,
    trainset, valset,
    adapter=RAGSystemPromptAdapter,
    reflection_lm=claude-opus-4-7,
    candidate_selection_strategy="pareto",
    max_metric_calls=100,
  )
        │
        ▼
  result.best_candidate["system_prompt"]
  → apply_best_prompt.py → formatters.py
```

---

## Dataset Preparation

**Sources:**
- 50 questions from `datasets/umass_data.entity.chatgpt.400.jsonl` (web-scraped reference docs)
- 50 questions from HotpotQA distractor split (2 gold + 8 distractor Wikipedia paragraphs)

Both sources bundle each question with its retrieved context documents — no FAISS index or embedding model required at preparation time.

**Split:** 100 combined questions, shuffled with seed=42
| Split | Size | Purpose |
|-------|------|---------|
| train | 60 (60%) | Evaluated during each GEPA mutation round |
| val   | 20 (20%) | Used by GEPA to rank candidates after each round |
| test  | 20 (20%) | Held out; used once to report final numbers |

Each JSONL line: `{"question": "...", "docs": [{<source fields>, "text": "..."}, ...]}`
- UMass docs: `{"url": "https://...", "text": "..."}`
- HotpotQA docs: `{"doc_id": "hotpot_X_Y", "url": "", "title": "...", "text": "..."}`

`title` and `doc_id` are metadata only — `references_to_documents()` generates its own `doc_id` and only propagates `url` + `text` into the live pipeline.

```python
@dataclass
class RAGDataInst:
    question: str
    docs: list[dict]   # source-dependent fields; always includes "text"
```

---

## Scoring Metrics

### Score 1 — Anti-Collapse (higher = more diverse = less collapse)

**Method:** Entity-based, matching `entity_extraction.py`.

```
1. Extract named entities from each round's answer via judge LLM
   (ENTITY_EXTRACTION_SYSTEM_PROMPT, temperature=0, one call per answer)

2. Cluster all surface mentions into canonical forms via judge LLM
   (ENTITY_CLUSTERING_SYSTEM_PROMPT, temperature=0, one call total)

3. Build binary entity-mention vector per answer
   vec[i] = 1.0 if canonical_entity_i appears in that answer's entity set

4. Compute pairwise cosine similarity across all answer vectors
   → mean_entity_similarity

5. anti_collapse = 1 − mean_entity_similarity
```

- Range: [0, 1]. Score near 1.0 means entity sets are diverse across rounds (no collapse). Score near 0.0 means all rounds mention the same entities (collapsed).
- Also returns `unique_entity_count` (total canonical entities across all rounds) — logged in the reflective dataset.
- **Threshold for diagnosis:** `anti_collapse < 0.30` → flags "Entity sets collapsed across rounds"

### Score 2 — Answer Quality (higher = better grounded)

Judge LLM evaluates the **final round's answer** against the **original uncontaminated context** (round 0 docs):

```
quality = 0.6 × faithfulness + 0.4 × relevance
```

- **Faithfulness:** Are all claims in the final answer supported by the original retrieved context? (1.0 = fully grounded, 0.0 = hallucinated/drifted)
- **Relevance:** Does the answer directly address the question? (1.0 = focused, 0.0 = off-topic)
- Faithfulness weighted higher because grounding is the primary concern.
- **Threshold for diagnosis:** `quality < 0.50` → flags "Final answer drifted from original context"
- Falls back to 0.5 on any parse error.

### Multi-Objective GEPA Return

```python
scores           = [avg_anti_collapse, ...]   # primary sort key
objective_scores = [{"anti_collapse": ..., "quality": ...}, ...]  # Pareto axes
```

GEPA maintains a Pareto frontier over `anti_collapse` and `quality`, ensuring the optimized prompt improves both rather than sacrificing one for the other.

---

## Adapter Implementation

**File:** `gepa_optimization/rag_adapter.py`

```
class RAGSystemPromptAdapter(GEPAAdapter):

    __init__(task_model, doc_gen_model, judge_model, api_base, api_key,
             n_rounds=10, embed_model="all-MiniLM-L6-v2", chars_per_doc=800)
        - task_llm:    ProprietaryLLM — runs RAG generation (the model being evaluated)
        - doc_gen_llm: ProprietaryLLM — generates AI contamination documents (separate)
        - judge_model: litellm string — entity extraction, clustering, quality judging
        - embed_model: local SentenceTransformer for simulate_search

    evaluate(batch, candidate, capture_traces) → EvaluationBatch
        For each RAGDataInst in batch:
            1. Build original_context from docs (saved as ground truth for quality judging)
            2. Run 3 collapse simulations (pipeline_simulator.py):
               - simulate_replace_all(question, refs, n_rounds=10, system_prompt, llm=doc_gen_llm)
               - simulate_replace_one(question, refs, n_rounds=10, system_prompt, llm=doc_gen_llm)
               - simulate_search(question, refs, n_rounds=10, system_prompt, llm=doc_gen_llm,
                                 embed_model=embed_model)
            3. Score each variant:
               - anti_collapse_score(question, answers, judge_model, ...) → (float, int)
               - judge_quality_score(question, original_context, final_answer, ...) → float
            4. Average anti_collapse and quality across 3 variants
            5. Return EvaluationBatch with scores, objective_scores, trajectories

    make_reflective_dataset(candidate, eval_batch, components_to_update)
        For each question trace, build a structured record with:
            - Inputs: question, context preview (600 chars), system prompt, n_rounds
            - Generated Outputs: Round 1..N answers per variant
            - Feedback: anti_collapse, unique_entities, quality, diagnosis per variant
            - Scores: avg_anti_collapse, avg_quality, avg_unique_entities
        This is what reflection_lm reads to propose better prompt mutations.
```

---

## Collapse Simulation Variants

**File:** `gepa_optimization/pipeline_simulator.py`

Uses real `pipeline/` context-building code. Only the LLM call is swapped to `ProprietaryLLM` (litellm/keymaker, no vLLM server needed).

| Variant | Context update rule | Collapse speed |
|---|---|---|
| `replace_all` | Entire context (10 docs) replaced by previous answer each round | Fastest |
| `replace_one` | One slot replaced per round (`slot = iteration % len(docs)`), starts with original refs capped at 10 | Slow decay |
| `search` | ChunkedRetrievalStore seeded with original refs; each round adds new answer and retrieves top-10 by cosine sim | Gradual contamination |

All variants run 10 rounds. The `search` variant uses `make_embed_fn_local("all-MiniLM-L6-v2")` — same embedding model as the production pipeline.

---

## GEPA Configuration

**File:** `gepa_optimization/run_optimization.py`

```python
result = gepa.optimize(
    seed_candidate={"system_prompt": _RAG_GENERATION_SYSTEM_PROMPT},
    trainset=trainset,    # 60 questions
    valset=valset,        # 20 questions
    adapter=adapter,
    reflection_lm=reflection_lm,          # claude-opus-4-7 via keymaker
    max_metric_calls=MAX_METRIC_CALLS,     # default 100, set via MAX_METRIC_CALLS env var
    candidate_selection_strategy="pareto",
    display_progress_bar=True,
    run_dir="./gepa_runs/rag_system_prompt",
)
```

| GEPA parameter | Value | Rationale |
|---|---|---|
| `candidate_selection_strategy` | `"pareto"` | Balances both objectives; prevents sacrificing quality for diversity |
| `max_metric_calls` | 100 (default) | Budget; increase for more thorough optimization |
| `run_dir` | `./gepa_runs/rag_system_prompt` | Checkpoints saved so job can resume if interrupted |

---

## Model Roles

| Env var | Default | Role |
|---|---|---|
| `TASK_MODEL` | `claude-haiku-4-5-20251001` | RAG generation in collapse simulations (the prompt being evaluated) |
| `DOC_GEN_MODEL` | `claude-haiku-4-5-20251001` | Generates AI contamination documents (separate from task model) |
| `JUDGE_MODEL` | `claude-haiku-4-5-20251001` | Entity extraction, entity clustering, quality judging |
| `REFLECTION_MODEL` | `claude-opus-4-7` | Reads failure reports and proposes new system prompts |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Local SentenceTransformer for `simulate_search` |
| `MAX_METRIC_CALLS` | `100` | GEPA evaluation budget |
| `API_KEY` | *(required)* | UMass keymaker API key |
| `LITELLM_API_BASE` | `https://thekeymaker.umass.edu/` | Proxy URL for all models |

---

## File Layout

```
gepa_optimization/
├── prepare_dataset.py        # Combines UMass + HotpotQA, splits 60/20/20
├── rag_adapter.py            # RAGSystemPromptAdapter (GEPAAdapter subclass)
├── scoring.py                # anti_collapse_score(), judge_quality_score()
├── pipeline_simulator.py     # simulate_replace_all/one/search
├── run_optimization.py       # Main GEPA entry point
├── apply_best_prompt.py      # Writes best_candidate back to formatters.py
└── data/
    ├── train.jsonl            # 60 questions
    ├── val.jsonl              # 20 questions
    └── test.jsonl             # 20 questions (held out)

gepa_runs/
└── rag_system_prompt/        # GEPA checkpoints (auto-created, resumable)
```

---

## How to Use the Result

After `gepa.optimize` finishes:

```python
# Best candidate is automatically selected from Pareto frontier (highest anti_collapse)
print(result.best_candidate["system_prompt"])
print(f"Best val score (avg anti_collapse): {result.best_score:.4f}")

# Apply to formatters.py
python gepa_optimization/apply_best_prompt.py --prompt '<optimized prompt>'
```

Then re-run evaluation on the held-out test split to verify improvement on:
- `unique_entities` (higher = more diverse content per round)
- `avg_pairwise_tes` (lower = less entity overlap = less collapse)
- `avg_ai_reference_percentage` (lower = less contamination)

The test split (20 questions) was never seen during optimization so these numbers are unbiased.

---

## Summary of Steps

1. **Run** `gepa_optimization/prepare_dataset.py` to generate `data/{train,val,test}.jsonl`
2. **Set** environment variables: `API_KEY`, and optionally `TASK_MODEL`, `DOC_GEN_MODEL`, `JUDGE_MODEL`, `REFLECTION_MODEL`, `MAX_METRIC_CALLS`
3. **Run** `gepa_optimization/run_optimization.py`
4. **Inspect** `gepa_runs/rag_system_prompt/` for checkpoints and Pareto frontier
5. **Apply** the best prompt: `python gepa_optimization/apply_best_prompt.py --prompt '<prompt>'`
6. **Evaluate** on the held-out test split using `evaluation.py`
