"""
Pipeline variant configuration: how many rounds we run and how many references we use.

We support four variants (set via --pipeline-variant). Two of them are "hybrid" in the
sense that context is built from both references and model-generated docs; the others are
retrieval-based.

  hybrid (pool mix)
    Each round the model gets (1) N synthetic docs (from its own past answers) and
    (2) M reference docs (from the dataset). You set N and M with --num-synth-docs and
    --num-db-docs. So you can do all synthetic (e.g. 10 synth, 0 db), some of each
    (e.g. 1 synth, 3 db), or any other combo—same pipeline, different knobs. We call it
    hybrid so we can extend it later to other replace-style experiments.

  replace_one (document setting) — under the hybrid idea
    Same goal as hybrid (context = refs + model generations), but a different rule: start
    with the question's references (up to 10). Each round, replace exactly one slot in
    that list with one new doc from the model's answer. The list evolves over 20 rounds.
    So we implement it as a separate variant because the update rule is different (one
    slot replaced in a fixed-length list vs. choosing N synth + M refs each round).

  search
    Not hybrid: each round we run vector search over all docs (original + generated so
    far) and feed the top results to the model. We run 30 rounds; no limit on refs.

  agentic_rag
    Same as search — vector store seeded from references, new AI-generated docs added each
    round — but instead of pre-fetching top-k into the prompt, the model is given a
    retrieve(query) tool and decides what to search on its own. 30 rounds.
"""

from typing import Final

# Variant identifiers (single source of truth; PIPELINE_VARIANTS is derived from this)
PIPELINE_HYBRID: Final[str] = "hybrid"
PIPELINE_REPLACE_ONE: Final[str] = "replace_one"
PIPELINE_SEARCH: Final[str] = "search"
PIPELINE_AGENTIC_RAG: Final[str] = "agentic_rag"

PIPELINE_VARIANTS: Final[tuple] = (PIPELINE_HYBRID, PIPELINE_REPLACE_ONE, PIPELINE_SEARCH, PIPELINE_AGENTIC_RAG)

# Rounds per variant (experiment defaults from paper)
ROUNDS_REPLACE_ALL: Final[int] = 10   # used for hybrid (any synth/db combo)
ROUNDS_REPLACE_ONE: Final[int] = 20
ROUNDS_SEARCH: Final[int] = 30

ROUNDS_BY_VARIANT: Final[dict] = {
    PIPELINE_HYBRID: ROUNDS_REPLACE_ALL,
    PIPELINE_REPLACE_ONE: ROUNDS_REPLACE_ONE,
    PIPELINE_SEARCH: ROUNDS_SEARCH,
    PIPELINE_AGENTIC_RAG: ROUNDS_SEARCH,
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

# Agentic RAG
AGENTIC_MAX_TOOL_CALLS: Final[int] = 10


def get_rounds_for_variant(variant: str) -> int:
    if variant not in ROUNDS_BY_VARIANT:
        raise ValueError(f"Unknown pipeline variant: {variant}. Use one of {PIPELINE_VARIANTS}")
    return ROUNDS_BY_VARIANT[variant]


def should_truncate_citations(variant: str) -> bool:
    """Replace One truncates at MAX_CITATIONS_REPLACE (paper: 10); Hybrid and Search do not."""
    return variant == PIPELINE_REPLACE_ONE


def is_replace_one(variant: str) -> bool:
    """True if variant is replace_one (paper: one slot replaced per round, evolving doc list)."""
    return variant == PIPELINE_REPLACE_ONE


def is_search(variant: str) -> bool:
    """True if variant is search (needs embed_fn, retrieval store)."""
    return variant == PIPELINE_SEARCH


def is_hybrid(variant: str) -> bool:
    """True if variant is hybrid (needs hybrid_config, initial_docs from refs)."""
    return variant == PIPELINE_HYBRID


def is_agentic_rag(variant: str) -> bool:
    """True if variant is agentic_rag (model retrieves context via tool calls)."""
    return variant == PIPELINE_AGENTIC_RAG
