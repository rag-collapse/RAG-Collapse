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

from typing import Dict, List

from formatters import (
    get_context_str_from_docs,
    get_rag_generation_conversation,
)


def build_rag_conversation(
    question: str,
    docs: List[Dict[str, str]],
    chars_per_doc: int,
) -> list[dict[str, str]]:
    """
    Build a RAG-style conversation using existing formatter utilities.

    This function intentionally contains no prompt logic of its own.
    """

    context = get_context_str_from_docs(
        docs=docs,
        chars_per_doc=chars_per_doc,
    )

    return get_rag_generation_conversation(
        context=context,
        question=question,
    )
