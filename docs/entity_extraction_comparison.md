# Entity Extraction: `RAG-Collapsement` vs. `collapse-randomness-research`

A focused compare-and-contrast of the **entity-extraction code** in the two repos.

| | This repo | Reference repo |
|---|---|---|
| File | `entity_extraction.py` | `randomness/core/metrics.py` |
| Form | Standalone CLI script, entities only | Function library inside a monolithic metrics module |
| Entity fns | `extract_entities_batch`, `cluster_entities`, `build_mention_to_canonical`, `recover_missed_entities`, `compute_entity_similarity`, `process_experiment_file` | `get_entities`, `dedup_entities_with_embeddings_map`, `fix_entities_map`, `retag_raw_entities`, `extract_entities`, `entity_similarity(_by_round)`, `get_entity_data`, `get_entities_by_round` |

> **Scope note.** In the reference repo, entity extraction lives *inside* `metrics.py` alongside citations, ranking (Kendall-τ), entropy, embeddings similarity, AI-retrieval, etc. This repo splits responsibilities: entity work in `entity_extraction.py`, text metrics (ROUGE/TES, AI-reference %, same-answer %) in `evaluation.py`. This doc compares only the entity-extraction portions.

---

## Pipeline, stage by stage

### 1. Extract entities from each response

| | This repo | Reference |
|---|---|---|
| Fn | `extract_entities_batch` | `get_entities` / `extract_entities` |
| Prompt input | **question + response** | **response only** (`ENTITIES_PROMPT` is just `{response}`) |
| Instruction | "extract only entities that **directly answer the question**" | "the main named entities that are **compared** … all of the same type" |
| LLM call | free-form text → `parse_json_from_response` (regex/JSON fallbacks) | `generate_openai_structured(..., NamedEntities)` — **schema-guaranteed** |
| Model | configurable (`gpt-4o` API default, or local vLLM) | pinned `gpt-5.2` |
| Concurrency | `llm.inference_batch` (vLLM batching), batch_size=20 | `multiprocessing.Pool`, one process per generation |
| On failure | returns `[]` (silent) | structured output → effectively can't malform |

**Contrast:** This repo is **question-conditioned** (more targeted, smaller entity sets) and tolerant of any backend, at the cost of relying on JSON parsing that can silently yield `[]`. The reference is **question-agnostic** (recall-oriented) and uses guaranteed-schema OpenAI output.

### 2. Cluster mentions → canonical entities

| | This repo | Reference |
|---|---|---|
| Fn | `cluster_entities` + `build_mention_to_canonical` | `dedup_entities_with_embeddings_map` + `fix_entities_map` |
| Output shape | dict `canonical → [mentions]`, then `mention.lower() → canonical` | dict `surface → representative` |
| Canonical label | LLM chooses the **most complete** name | **shortest** mention wins (`sorted by (len, e)`) |
| Coverage guard | fallback "each mention is its own cluster" if parse fails | `fix_entities_map` re-checks every input mention, fixes casing, prints `NOT FOUND` warnings |
| Despite the name | — | `..._with_embeddings_map` does **not** use embeddings; it's pure LLM grouping |

**Contrast:** Both cluster via LLM. The reference actively **reconciles dropped/мis-cased mentions** (`fix_entities_map`) so every input mention lands in the map; this repo instead lets an unmapped mention become its own canonical at use-time (`process_experiment_file`, ~line 338), which can leave case-variant near-duplicate canonicals. Opposite label heuristics (shortest vs. most-complete).

### 3. Recover entities the extractor missed

| | This repo | Reference |
|---|---|---|
| Fn | `recover_missed_entities` | `retag_raw_entities` |
| Match | **case-insensitive** substring (`mention in response_lower`) | **case-sensitive** substring (`k in generation`) |
| Effect | adds canonical if its mention appears in the text but extractor missed it | same idea, per round |

**Contrast:** Same recall-boosting intent; this repo's case-insensitive match recovers strictly more. The reference additionally builds **positional** info for ranking (below); this repo does not.

### 4. Per-round aggregation

| | This repo | Reference |
|---|---|---|
| "Unique entities" | `len(canonical_in_round)` — distinct canonical entities seen across runs that round | `len(entity_counts)` from `get_entity_data` — distinct mapped entities that round |
| Both | count of distinct canonical entities per round | same concept |

Equivalent in spirit.

### 5. Entity-similarity metric ⚠️ key difference

| | This repo (`compute_entity_similarity`) | Reference (`entity_similarity`) |
|---|---|---|
| Vector | binary over canonical alphabet (`to_vec` equiv.) | binary over canonical alphabet (`to_vec`) |
| Cosine | manual `dot/(n1*n2)` | scipy `1 - distance.cosine` |
| **Empty response** | norm 0 → **`0.0`** (dissimilar) | either vector all-zero → **`1.0`** (identical) |
| Reported stats | mean / min / max / std | mean only (per round) |
| n < 2 | returns zeros | n/a (pairwise over combinations) |

**This is the single most consequential difference.** When a degenerated response contains **no entities**, the reference treats the pair as *fully collapsed* (1.0) while this repo treats it as *fully diverse* (0.0). The two conventions push the collapse curve in **opposite directions**, so any run with empty extractions will not be comparable across the two codebases unless this is aligned.

---

## Entity-related capabilities only in the reference

- **Ranking similarity (Kendall-τ):** `get_ranking` + `_find_entity_position` + `_find_words_in_order` + `ranked_list_similarity`. Orders entities by their **position** in the response, with exact → alias → case-insensitive → "words-in-order-with-gaps" matching, then compares orderings with Kendall-τ. This repo has **no positional/ranking metric** and no fuzzy position finder.
- **Entity entropy:** `get_count_entropy` over the per-round entity count distribution. Absent here.
- **AI entity influence:** `input_to_generation_entity_similarity(_by_round)` — how much generated answers' entities track AI-authored input pages vs. original pages. Absent here.
- **Cost tracking:** reference threads `increment_cost`; this repo does not.

(This repo computes ROUGE-1/2/L and TES in `evaluation.py`, which the reference's `metrics.py` does not — it uses embedding cosine "mean similarity" instead. So divergence runs both ways, but those are outside entity extraction.)

---

## Summary

| Dimension | This repo | Reference | Comparable? |
|---|---|---|---|
| Extraction conditioning | question + response | response only | ❌ different entity sets |
| LLM output contract | parsed free-form (any backend) | structured schema (gpt-5.2) | ⚠️ reliability differs |
| Canonical label | most-complete | shortest | cosmetic |
| Unmapped mentions | becomes own canonical | reconciled by `fix_entities_map` | ⚠️ dup risk here |
| Missed-entity recovery | case-insensitive substring | case-sensitive substring | ✅ similar (this repo recovers more) |
| Unique-entities-per-round | distinct canonical / round | distinct canonical / round | ✅ equivalent |
| Empty-response similarity | `0.0` (diverse) | `1.0` (collapsed) | ❌ **opposite — flips collapse signal** |
| Ranking (Kendall-τ), entropy, AI-influence | absent | present | ❌ missing here |

### If the goal is comparability with the reference's published numbers
1. **Align the empty-response similarity convention** (`compute_entity_similarity`, the `else 0.0` branch) — this is the one that can change conclusions.
2. **Decide on question-conditioning** in extraction — keep targeted, or drop the question to match the reference's recall-oriented set; document the choice in Methods.
3. Optionally port **Kendall-τ ranking similarity** and **AI entity influence** to close the metric gap.

### Where this repo is arguably better
- Backend-agnostic (runs local vLLM models, not just OpenAI), batched inference.
- Case-insensitive recovery catches more missed mentions.
- Richer similarity stats (min/max/std, not just mean).
