import copy
from typing import Any, Dict, List


def references_to_documents(
    references: List[Dict[str, Any]],
    iteration: int = 0,
) -> List[Dict[str, Any]]:
    """
    Convert dataset references into documents for the pipeline.

    Expected input: a list of dicts like {"url": "...", "text": "...", ...}
    Output docs include stable ids + iteration metadata.
    """
    docs: List[Dict[str, Any]] = []
    for i, r in enumerate(references or []):
        docs.append(
            {
                "doc_id": f"ref_{iteration}_{i}",
                "iteration": iteration,
                "url": r.get("url", ""),
                "text": r.get("text", ""),
            }
        )
    return docs


def answers_to_documents(
    answers: List[str],
    iteration: int,
) -> List[Dict[str, Any]]:
    """
    Convert model answers into documents for the next iteration.
    """
    docs: List[Dict[str, Any]] = []
    for i, a in enumerate(answers or []):
        docs.append(
            {
                "doc_id": f"gen_{iteration}_{i}",
                "iteration": iteration,
                "url": "model_generated",
                "text": a,
            }
        )
    return docs


def next_docs_replace_one(
    current_docs: List[Dict[str, Any]],
    new_document_texts: List[str],
    iteration: int,
) -> List[Dict[str, Any]]:
    """
    Replace One: replace one slot per round with one new AI-generated doc.
    Slot index = iteration % len(current_docs). Evolving doc list over rounds.
    """
    if not new_document_texts or not current_docs:
        return current_docs
    slot = iteration % len(current_docs)
    next_docs = copy.deepcopy(current_docs)
    one_new_doc = answers_to_documents([new_document_texts[0]], iteration=iteration)[0]
    next_docs[slot] = one_new_doc
    return next_docs


def inject_generated_text(
    example: Dict[str, Any],
    generated_text: str,
) -> Dict[str, Any]:
    """
    Optional helper (not used by run_pipeline.py):
    Create a new example where model-generated text is injected back as context.
    """
    return {
        "question": example["question"],
        "references": [{"url": "model_generated", "text": generated_text}],
        "original_references": example.get("references", []),
    }
