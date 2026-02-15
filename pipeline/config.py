"""
Pipeline variant configuration: round caps, citation limits, and defaults.

- Hybrid: one pipeline for "replace"-style behavior. Context each round = num_synth_docs
  (from model generations) + num_db_docs (from this question's references). Configurable
  via --num-synth-docs, --num-db-docs, --db-doc-selection, --synth-doc-selection.
  E.g. replace_all-style = --num-synth-docs 10 --num-db-docs 0; fixed mix = --num-synth-docs 1 --num-db-docs 3.
- Search: vector retrieval each round (LiteLLM embeddings); 30 rounds; no citation cap.
"""

from typing import Final

# Variant identifiers (single source of truth; PIPELINE_VARIANTS is derived from this)
PIPELINE_HYBRID: Final[str] = "hybrid"
PIPELINE_SEARCH: Final[str] = "search"

PIPELINE_VARIANTS: Final[tuple] = (PIPELINE_HYBRID, PIPELINE_SEARCH)

# Rounds per variant (single lookup table; no if-chain)
ROUNDS_BY_VARIANT: Final[dict] = {
    PIPELINE_HYBRID: 10,
    PIPELINE_SEARCH: 30,
}

# Minimum number of references (citations) per question; questions with fewer are skipped
MIN_CITATIONS: Final[int] = 5

# Optional cap on references per question (used only if should_truncate_citations; currently False)
MAX_CITATIONS_REPLACE: Final[int] = 10

# Runs per round (responses per question per round)
RUNS_PER_ROUND: Final[int] = 10

# Search retrieval
SEARCH_TOP_K: Final[int] = 10
SEARCH_CHUNK_SIZE: Final[int] = 500
SEARCH_CHUNK_OVERLAP: Final[int] = 50


def get_rounds_for_variant(variant: str) -> int:
    if variant not in ROUNDS_BY_VARIANT:
        raise ValueError(f"Unknown pipeline variant: {variant}. Use one of {PIPELINE_VARIANTS}")
    return ROUNDS_BY_VARIANT[variant]


def should_truncate_citations(variant: str) -> bool:
    """Hybrid and Search do not truncate references (we need full pool for configurable mix)."""
    return False


def is_search(variant: str) -> bool:
    """True if variant is search (needs embed_fn, retrieval store)."""
    return variant == PIPELINE_SEARCH


def is_hybrid(variant: str) -> bool:
    """True if variant is hybrid (needs hybrid_config, initial_docs from refs)."""
    return variant == PIPELINE_HYBRID
