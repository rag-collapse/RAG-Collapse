"""
Pipeline variant configuration: round caps, citation limits, and defaults.

- Hybrid: one pipeline for "replace"-style behavior. Context each round = num_synth_docs
  (from model generations) + num_db_docs (from this question's references). Configurable
  via --num-synth-docs, --num-db-docs, --db-doc-selection, --synth-doc-selection.
  E.g. replace_all-style = --num-synth-docs 10 --num-db-docs 0; fixed mix = --num-synth-docs 1 --num-db-docs 3.
- Search: vector retrieval each round (LiteLLM embeddings); 30 rounds; no citation cap.
"""

from typing import Final

# Variant identifiers
PIPELINE_HYBRID: Final[str] = "hybrid"
PIPELINE_SEARCH: Final[str] = "search"

PIPELINE_VARIANTS: Final[tuple] = (PIPELINE_HYBRID, PIPELINE_SEARCH)

# Minimum number of references (citations) per question; questions with fewer are skipped
MIN_CITATIONS: Final[int] = 5

# Optional cap on references per question (used only if should_truncate_citations; currently False)
MAX_CITATIONS_REPLACE: Final[int] = 10

# Rounds per variant
ROUNDS_SEARCH: Final[int] = 30
ROUNDS_HYBRID: Final[int] = 10

# Runs per round (responses per question per round)
RUNS_PER_ROUND: Final[int] = 10

# Search retrieval
SEARCH_TOP_K: Final[int] = 10
SEARCH_CHUNK_SIZE: Final[int] = 500
SEARCH_CHUNK_OVERLAP: Final[int] = 50


def get_rounds_for_variant(variant: str) -> int:
    if variant == PIPELINE_HYBRID:
        return ROUNDS_HYBRID
    if variant == PIPELINE_SEARCH:
        return ROUNDS_SEARCH
    raise ValueError(f"Unknown pipeline variant: {variant}. Use one of {PIPELINE_VARIANTS}")


def should_truncate_citations(variant: str) -> bool:
    """Hybrid and Search do not truncate references (we need full pool for configurable mix)."""
    return False
