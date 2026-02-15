"""
Context builder: single place that computes the document list for the next iteration.

Two variants: hybrid (configurable ratio of synthetic vs original refs) and search.
Hybrid subsumes replace_all- and replace_one-style behavior via --num-synth-docs / --num-db-docs.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pipeline.config import PIPELINE_VARIANTS, is_hybrid, is_search, SEARCH_TOP_K
from pipeline.feedback_loop import answers_to_documents, references_to_documents


@dataclass(frozen=True)
class HybridContextConfig:
    """Config for the hybrid variant: fixed mix of synthetic + database docs."""

    num_synth_docs: int
    num_db_docs: int
    db_doc_selection: str  # "first" | "random"
    synth_doc_selection: str  # "first" | "random"


def _select_items(items: List[Any], k: int, mode: str) -> List[Any]:
    """Select up to k items: mode 'first' = items[:k], 'random' = random.sample."""
    if mode == "first":
        return list(items[:k])
    if k >= len(items):
        return list(items)
    return list(random.sample(items, k))


def _hybrid_next_docs(
    document_texts: List[str],
    references: List[Dict[str, Any]],
    iteration: int,
    config: HybridContextConfig,
) -> List[Dict[str, Any]]:
    """
    Build next iteration context: num_synth_docs from model outputs + num_db_docs
    from this question's references. No cross-question leakage; refs are per-question.
    """
    selected_texts = _select_items(
        document_texts, config.num_synth_docs, config.synth_doc_selection
    )
    synth_docs = answers_to_documents(selected_texts, iteration=iteration) if selected_texts else []

    selected_refs = _select_items(references, config.num_db_docs, config.db_doc_selection)
    db_docs = references_to_documents(selected_refs, iteration=0) if selected_refs else []

    return synth_docs + db_docs


def get_next_documents(
    variant: str,
    current_docs: List[Dict[str, Any]],
    document_texts: List[str],
    iteration: int,
    *,
    references: Optional[List[Dict[str, Any]]] = None,
    question_text: Optional[str] = None,
    store: Optional[Any] = None,
    hybrid_config: Optional[HybridContextConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Return the document list for the next iteration.

    - hybrid: num_synth_docs from document_texts + num_db_docs from references (this question).
      With num_db_docs=0 you get replace_all-style (all synth); with num_synth_docs=1, num_db_docs=3 you get a fixed mix.
    - search: new docs added to store; return top-k retrieval for question.
    """
    if is_hybrid(variant):
        if hybrid_config is None or references is None:
            raise ValueError("hybrid variant requires hybrid_config and references")
        return _hybrid_next_docs(
            document_texts,
            references,
            iteration=iteration,
            config=hybrid_config,
        )

    if is_search(variant):
        if store is None or question_text is None:
            raise ValueError("search variant requires store and question_text")
        new_docs = answers_to_documents(document_texts, iteration=iteration)
        store.add_documents(new_docs)
        return store.search(question_text, k=SEARCH_TOP_K)

    raise ValueError(f"Unknown variant: {variant}. Use one of {PIPELINE_VARIANTS}")


def get_initial_documents_hybrid(
    references: List[Dict[str, Any]],
    config: HybridContextConfig,
) -> List[Dict[str, Any]]:
    """
    Initial context for hybrid variant (iteration 0): no synthetic docs yet.
    If num_db_docs=0 (replace_all-style), use all references so round 0 has context.
    Otherwise use num_db_docs refs with db_doc_selection.
    """
    if config.num_db_docs == 0:
        return references_to_documents(references, iteration=0)
    selected_refs = _select_items(references, config.num_db_docs, config.db_doc_selection)
    return references_to_documents(selected_refs, iteration=0) if selected_refs else []
