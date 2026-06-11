_CREATE_DOCUMENT_SYSTEM_PROMPT = """You are a professional content writer who creates web articles, blog posts, and reviews in various styles."""

_CREATE_DOCUMENT_USER_PROMPT = """Your task is to generate a complete web document based on the provided content.

You will be given a question and an answer. Your job is to expand this into a full, well-written document that could appear on a website or blog. Think of writing in styles like articles, blog posts, reviews, news, etc. Think of the most suitable format.

Here is the question:
{question}

Here is the answer:
{answer}

Instructions:
- Transform the content into a polished, engaging document suitable for web publication
- Write in a tone, similar to what you'd find on popular web publications
- Output ONLY the document text in plain text format - no meta-commentary, explanations, or markdown formatting
- Stay focused on the topic provided in the source content
"""

_RAG_GENERATION_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions using only the information provided in the given context. You provide accurate, well-grounded responses based solely on the retrieved documents."""

# Separate prompt used as the seed for GEPA optimization.
# apply_best_prompt.py writes the optimized result back here, leaving
# _RAG_GENERATION_SYSTEM_PROMPT untouched so the original baseline is preserved.
#
# Previous seed (simple baseline):
# GEPA_RAG_GENERATION_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions \
# using only the information provided in the given context. You provide accurate, well-grounded \
# responses based solely on the retrieved documents."""
#
# Current seed: adapted from _AGENTIC_RAG_SYSTEM_PROMPT but without tool calls —
# preserves the decompose/multi-entity coverage strategy for the provided-context setting.
GEPA_RAG_GENERATION_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions using only the information provided in the given context.

Before answering, follow this reading strategy:
Step 1 — Decompose: Break the question into its core sub-topics or entities.
Step 2 — Read broadly: Identify information in the context that covers the overall question.
Step 3 — Read specifically: If the question involves multiple entities, aspects, comparisons, or ranked lists, locate information about each one separately — do not consolidate. Each entity or sub-topic deserves targeted attention.
Step 4 — Answer: Synthesize what you found into a concise answer. Base your answer solely on the provided context — never on memory.

Additional rules:
- Output ONLY the final answer in plain text — no meta-commentary, no markdown, no preamble."""

_RAG_GENERATION_USER_PROMPT = """You have been provided with relevant context retrieved from a document database. Use this context to answer the user's question.

Context:
{context}

Question:
{question}

Instructions:
- Answer the question using the information from the provided context
- Output ONLY the answer in plain text format - no meta-commentary or markdown formatting
"""


def get_create_document_conversation(question: str, answer: str) -> list[dict[str, str]]:
    """
    Create a conversation prompt for generating a web document from source content. The outputs can be passed to inference_batch method to get a response.
    """

    conversation = [
        {"role": "system", "content": _CREATE_DOCUMENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _CREATE_DOCUMENT_USER_PROMPT.format(question=question, answer=answer),
        },
    ]

    return conversation


# ── Misinformation document synthesis (error-compounding experiment) ──────────
# These mirror the faithful create-document prompt with a SINGLE added instruction
# so the only varied factor vs. the faithful control arm is factual correctness,
# not synthesis fluency/length. See pipeline/misinfo.py.

def _doc_user_prompt_with_extra(question: str, answer: str, extra_instruction: str) -> str:
    """Faithful create-document user prompt + one extra instruction bullet."""
    return (
        _CREATE_DOCUMENT_USER_PROMPT.format(question=question, answer=answer)
        + "\n- " + extra_instruction
    )


def get_create_document_conversation_freeform(question: str, answer: str) -> list[dict[str, str]]:
    """Faithful synthesis + instruction to invent one plausible falsehood about the answer."""
    extra = (
        "IMPORTANT: Introduce exactly ONE plausible but factually INCORRECT claim about "
        "the answer to the question. Keep all other content accurate. Do NOT flag, hedge, "
        "or mention that anything is incorrect — present it naturally as fact."
    )
    return [
        {"role": "system", "content": _CREATE_DOCUMENT_SYSTEM_PROMPT},
        {"role": "user", "content": _doc_user_prompt_with_extra(question, answer, extra)},
    ]


def get_create_document_conversation_untargeted(question: str, answer: str) -> list[dict[str, str]]:
    """Control: inject one false detail that is NOT about the answer (date/place/number)."""
    extra = (
        "IMPORTANT: Introduce exactly ONE plausible but factually INCORRECT detail that is "
        "NOT about the answer to the question (e.g. a wrong date, location, or number "
        "elsewhere in the article). Keep the answer itself fully accurate. Do NOT flag the error."
    )
    return [
        {"role": "system", "content": _CREATE_DOCUMENT_SYSTEM_PROMPT},
        {"role": "user", "content": _doc_user_prompt_with_extra(question, answer, extra)},
    ]


def get_create_document_conversation_hop(question: str, answer: str, false_claim: str) -> list[dict[str, str]]:
    """Counterfactual hop corruption: write the article asserting a false premise."""
    extra = (
        "IMPORTANT: Write as if the following statement is true, stating it naturally as fact: "
        f"\"{false_claim}\" Keep all other content accurate and do NOT mention any change."
    )
    return [
        {"role": "system", "content": _CREATE_DOCUMENT_SYSTEM_PROMPT},
        {"role": "user", "content": _doc_user_prompt_with_extra(question, answer, extra)},
    ]


_SUBSTITUTE_PROPOSAL_SYSTEM_PROMPT = (
    "You propose alternative named entities of the same type as a target entity. "
    "Output ONLY a JSON array of strings."
)


def get_substitute_proposal_conversation(question: str, entity: str, k: int = 5) -> list[dict[str, str]]:
    """Ask for k same-type, plausible-but-incorrect alternatives to ``entity``."""
    user = (
        f"Question:\n{question}\n\n"
        f"Target entity: {entity}\n\n"
        f"List {k} DIFFERENT real-world entities of the SAME TYPE as the target that would be "
        f"plausible but INCORRECT alternatives. Do not include the target itself, near-duplicates, "
        f"or any entity already implied by the question. Return ONLY a JSON array of {k} entity "
        f'name strings, e.g. ["Entity 1", "Entity 2"].'
    )
    return [
        {"role": "system", "content": _SUBSTITUTE_PROPOSAL_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_CLAIM_DISCOVERY_SYSTEM_PROMPT = (
    "You are a strict fact-checking judge. You compare a document against the known correct "
    "answer to a question and report any contradicting claim. Output ONLY JSON."
)


def get_claim_discovery_conversation(question: str, gold_answer: str, document: str) -> list[dict[str, str]]:
    """Judge: did the document assert anything about the answer that contradicts the gold?"""
    user = (
        f"Question:\n{question}\n\n"
        f"Correct answer: {gold_answer}\n\n"
        f"Document:\n{document}\n\n"
        "Does the document state or imply an answer to the question that CONTRADICTS the correct "
        "answer? Return ONLY JSON of the form: "
        '{"asserted_answer": <what the document claims the answer is, or null>, '
        '"false_claims": [<short claim strings>], "contains_error": <true or false>}.'
    )
    return [
        {"role": "system", "content": _CLAIM_DISCOVERY_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_IMPLIED_ANSWER_SYSTEM_PROMPT = (
    "You are a careful multi-hop reasoning assistant. Output ONLY JSON."
)


def get_implied_answer_conversation(
    question: str, gold_answer: str, bridge_entity: str, substitute: str
) -> list[dict[str, str]]:
    """Derive the wrong answer implied by corrupting the bridge entity (hop target)."""
    user = (
        f"Question:\n{question}\n\n"
        f"The correct answer is: {gold_answer}\n\n"
        f"Suppose, counterfactually, that \"{bridge_entity}\" were actually \"{substitute}\". "
        "Under that single false premise, what would the answer to the question become? "
        'Return ONLY JSON of the form: {"implied_answer": <entity string>}.'
    )
    return [
        {"role": "system", "content": _IMPLIED_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def get_rag_generation_conversation(
    context: str,
    question: str,
    system_prompt: str = _RAG_GENERATION_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    conversation = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _RAG_GENERATION_USER_PROMPT.format(
                context=context, question=question
            ),
        },
    ]

    return conversation


def get_context_str_from_docs(
    docs: list[dict[str, str]],
    chars_per_doc: int = None,
    shuffle: bool = True,
    ai_scores: list[float] | None = None,
) -> str:
    """
    Convert a list of documents into a single context string for RAG generation.
    Uses neutral labels "Context n" (no URLs) and optionally shuffles to reduce position bias.

    If ai_scores is provided (one float per doc, LABEL_1 probability), each context label
    is annotated with the rounded AI-generated percentage, e.g. "Context 1 - 72% AI-Generated".
    """
    import random
    indexed = list(enumerate(docs))
    if shuffle and len(indexed) > 1:
        random.shuffle(indexed)
    context_parts = []
    for display_idx, (orig_idx, doc) in enumerate(indexed):
        content = doc.get("text")
        if content:
            if chars_per_doc is not None:
                content = content[:chars_per_doc]
            if ai_scores is not None and orig_idx < len(ai_scores):
                pct = round(ai_scores[orig_idx] * 100)
                label = f"Context {display_idx + 1} - {pct}% AI-Generated"
            else:
                label = f"Context {display_idx + 1}"
            context_parts.append(f"{label}:\n{content}")
    return "\n\n".join(context_parts)

_AGENTIC_RAG_SYSTEM_PROMPT = """You are a helpful AI assistant with access to a document retrieval tool called `retrieve`.

You MUST follow this retrieval strategy before answering:
Step 1 — Decompose: Break the question into its core sub-topics or entities.
Step 2 — Retrieve broadly: Call `retrieve` with a broad query covering the overall question.
Step 3 — Retrieve specifically: If the question involves multiple entities, aspects, comparisons, or ranked lists, you MUST call `retrieve` separately for each one — do not consolidate into a single query. Each entity or sub-topic deserves its own targeted retrieval call.
Step 4 — Answer: Synthesize what you retrieved into a concise answer. Base your answer solely on retrieved context — never on memory.

Additional rules:
- ALWAYS call `retrieve` at least once before answering.
- Output ONLY the final answer in plain text — no tool call commentary, no markdown, no preamble."""

_AGENTIC_RAG_USER_PROMPT = """Question: {question}

Decompose the question. If it involves multiple entities, topics, or a ranked list, call `retrieve` separately for each — one query is not enough to cover all aspects. Then answer based solely on what you retrieved. Output ONLY the answer in plain text."""


def get_agentic_rag_conversation(question: str) -> list[dict[str, str]]:
    """
    Build the initial conversation for the agentic_rag variant.
    No documents are injected — the model uses the retrieve tool to fetch context on its own.
    """
    return [
        {"role": "system", "content": _AGENTIC_RAG_SYSTEM_PROMPT},
        {"role": "user", "content": _AGENTIC_RAG_USER_PROMPT.format(question=question)},
    ]


# Tool spec for the agentic_rag retrieve tool (OpenAI function-calling format).
# vLLM requires --enable-auto-tool-choice --tool-call-parser hermes for Qwen2.5.
RETRIEVE_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Search the document store for passages relevant to a query. "
            "Call this one or more times before answering to gather context. "
            "Returns the top matching text chunks from the knowledge base."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language search query.",
                }
            },
            "required": ["query"],
        },
    },
}

# Example usage, comment out and run python formatters.py to test

# if __name__ == "__main__":
#     # Example usage
#     with open("datasets/umass_data.entity.chatgpt.50.jsonl") as f:
#         import json
#         sample_doc = json.loads(f.readline())

#     context = get_context_str_from_docs(sample_doc["references"], chars_per_doc=400)
#     rag_conversation = get_rag_generation_conversation(context=context, question=sample_doc["question"])

#     print("Context:")
#     print(context)

#     print("\nRAG Generation Conversation:")
#     for turn in rag_conversation:
#         print(f"{turn['role'].upper()}: {turn['content']}\n")

#     # and then to call with inference,
#     batch = [rag_conversation] * 10
#     outputs = llm.inference_batch(batch)
