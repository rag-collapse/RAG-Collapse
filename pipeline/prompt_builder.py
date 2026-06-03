# from typing import Dict, List

# from formatters import (
#     _RAG_GENERATION_SYSTEM_PROMPT,
#     _RAG_GENERATION_USER_PROMPT,
# )


# def build_prompt(example: Dict) -> Dict[str, str]:
#     """
#     Build a prompt for a single example using existing formatter templates.

#     This function does NOT modify reference text.
#     It only formats it into a prompt structure.
#     """
#     question = example["question"]
#     references: List[Dict] = example["references"]

#     context_text = "\n\n".join(ref["text"] for ref in references)

#     return {
#         "system": _RAG_GENERATION_SYSTEM_PROMPT,
#         "user": _RAG_GENERATION_USER_PROMPT.format(
#             question=question,
#             context=context_text,
#         ),
#     }

from typing import Dict, List, Optional

from formatters import (
    get_context_str_from_docs,
    get_rag_generation_conversation,
    get_agentic_rag_conversation,
)


def build_rag_conversation(
    question: str,
    docs: List[Dict[str, str]],
    chars_per_doc: int,
    shuffle_docs: bool = True,
    ai_scores: Optional[List[float]] = None,
    system_prompt: Optional[str] = None,
) -> list[dict[str, str]]:
    """
    Build a RAG-style conversation using existing formatter utilities.
    Docs are shown as "Context n" and shuffled each round to reduce position bias.

    If ai_scores is provided (one float per doc, LABEL_1 probability), each context
    label is annotated with the AI-generated percentage.

    If system_prompt is provided it overrides the default _RAG_GENERATION_SYSTEM_PROMPT
    (e.g. pass GEPA_RAG_GENERATION_SYSTEM_PROMPT for optimized-prompt runs).
    """

    context = get_context_str_from_docs(
        docs=docs,
        chars_per_doc=chars_per_doc,
        shuffle=shuffle_docs,
        ai_scores=ai_scores,
    )

    kwargs = {} if system_prompt is None else {"system_prompt": system_prompt}
    return get_rag_generation_conversation(
        context=context,
        question=question,
        **kwargs,
    )


def build_agentic_rag_conversation(question: str) -> list[dict[str, str]]:
    """
    Build the initial conversation for the agentic_rag variant.
    No documents are pre-injected; the model calls the retrieve tool autonomously.
    """
    return get_agentic_rag_conversation(question)
