"""
Simulates the three RAG collapse scenarios using the real pipeline code.

Imports directly from pipeline/, formatters.py, and llm_service/ so the
context-building, document-management, and retrieval logic is identical to
what runs in production. Only the LLM call is swapped: ProprietaryLLM
(litellm-backed, API mode) replaces ServerLLM so no vLLM server is needed.

Variants
--------
replace_all  — hybrid config with num_synth=all, num_db=0.
               Round 0: original reference docs.
               Each subsequent round: entire context is replaced by the
               model's previous answer. Fastest-collapsing scenario.

replace_one  — Paper variant.
               Starts with original refs (≤MAX_CITATIONS_REPLACE=10).
               Each round exactly one slot is replaced by the latest answer
               (slot = iteration % len(current_docs)).

search       — ChunkedRetrievalStore (litellm embeddings, keymaker proxy).
               Seeded with original refs; each round adds the new answer and
               retrieves top-k by cosine similarity. As AI text dominates the
               store the model increasingly echoes itself.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.feedback_loop import references_to_documents, answers_to_documents, next_docs_replace_one
from pipeline.context_builder import (
    HybridContextConfig,
    get_next_documents,
    get_initial_documents_hybrid,
    get_initial_documents_replace_one,
)
from pipeline.config import SEARCH_TOP_K, MAX_CITATIONS_REPLACE
from pipeline.retrieval import ChunkedRetrievalStore, make_embed_fn_local
from formatters import get_context_str_from_docs, get_rag_generation_conversation

# replace_all is hybrid with all-synth, no-db docs
_REPLACE_ALL_CONFIG = HybridContextConfig(
    num_synth_docs=10,
    num_db_docs=0,
    db_doc_selection="first",
    synth_doc_selection="first",
)


def _generate_one(conversation: list[dict], llm) -> str:
    """Single inference call via ProprietaryLLM (litellm-backed)."""
    try:
        results = llm.inference_batch([conversation])
        return (results[0] or "").strip()
    except Exception:
        return ""


# ── replace_all ───────────────────────────────────────────────────────────────

def simulate_replace_all(
    question: str,
    refs: list[dict],
    n_rounds: int,
    system_prompt: str,
    llm,
) -> list[str]:
    """
    Uses real hybrid context building (num_synth=10, num_db=0).
    Round 0: original refs. Round N+1: model's answer from round N only.
    """
    current_docs = get_initial_documents_hybrid(refs, _REPLACE_ALL_CONFIG)
    answers = []

    for round_idx in range(n_rounds):
        context = get_context_str_from_docs(current_docs, shuffle=True)
        conversation = get_rag_generation_conversation(context, question, system_prompt=system_prompt)
        answer = _generate_one(conversation, llm)
        answers.append(answer)

        # Next round: ALL context replaced with this answer
        current_docs = get_next_documents(
            variant="hybrid",
            current_docs=current_docs,
            document_texts=[answer] if answer else [],
            iteration=round_idx + 1,
            references=refs,
            hybrid_config=_REPLACE_ALL_CONFIG,
        )

    return answers


# ── replace_one ───────────────────────────────────────────────────────────────

def simulate_replace_one(
    question: str,
    refs: list[dict],
    n_rounds: int,
    system_prompt: str,
    llm,
) -> list[str]:
    """
    Uses real next_docs_replace_one via get_next_documents.
    Starts with real refs (≤10); one slot replaced per round.
    """
    current_docs = get_initial_documents_replace_one(refs)[:MAX_CITATIONS_REPLACE]
    answers = []

    for round_idx in range(n_rounds):
        context = get_context_str_from_docs(current_docs, shuffle=True)
        conversation = get_rag_generation_conversation(context, question, system_prompt=system_prompt)
        answer = _generate_one(conversation, llm)
        answers.append(answer)

        current_docs = get_next_documents(
            variant="replace_one",
            current_docs=current_docs,
            document_texts=[answer] if answer else [],
            iteration=round_idx + 1,
        )

    return answers


# ── search ───────────────────────────────────────────────────────────────────

def simulate_search(
    question: str,
    refs: list[dict],
    n_rounds: int,
    system_prompt: str,
    llm,
    embed_model: str = "all-MiniLM-L6-v2",
) -> list[str]:
    """
    Uses real ChunkedRetrievalStore with local SentenceTransformer embeddings.
    Seeded with original refs; adds each answer and retrieves top-k each round.
    """
    embed_fn = make_embed_fn_local(model_name=embed_model)
    store = ChunkedRetrievalStore(embed_fn=embed_fn)

    # Seed store with original references
    initial_docs = references_to_documents(refs, iteration=0)
    store.add_documents(initial_docs)

    answers = []

    for round_idx in range(n_rounds):
        retrieved = get_next_documents(
            variant="search",
            current_docs=[],
            document_texts=[],
            iteration=round_idx,
            store=store,
            question_text=question,
            search_top_k=SEARCH_TOP_K,
        )
        context = get_context_str_from_docs(retrieved, shuffle=False)
        conversation = get_rag_generation_conversation(context, question, system_prompt=system_prompt)
        answer = _generate_one(conversation, llm)
        answers.append(answer)

        # Add answer to store for next round
        new_docs = answers_to_documents([answer] if answer else [], iteration=round_idx + 1)
        store.add_documents(new_docs)

    return answers
