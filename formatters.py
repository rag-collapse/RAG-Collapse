_CREATE_DOCUMENT_SYSTEM_PROMPT = """You are a professional content writer who creates web articles, blog posts, and reviews in various styles."""

_CREATE_DOCUMENT_USER_PROMPT = """Your task is to generate a complete web document based on the provided source content.

You will be given source content about a topic. Your job is to expand this into a full, well-written document that could appear on a website or blog. Think of writing in styles like articles, blog posts, reviews, news, etc.

Here is the source content to base your document on:
{content}

Instructions:
- Transform the source content into a polished, engaging document suitable for web publication
- Write in a tone, similar to what you'd find on popular web publications
- Output ONLY the document text in plain text format - no meta-commentary, explanations, or markdown formatting
- Stay focused on the topic provided in the source content
"""

_RAG_GENERATION_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions using only the information provided in the given context. You provide accurate, well-grounded responses based solely on the retrieved documents."""

_RAG_GENERATION_USER_PROMPT = """You have been provided with relevant context retrieved from a document database. Use this context to answer the user's question.

Context:
{context}

Question:
{question}

Instructions:
- Answer the question using the information from the provided context
- Output ONLY the answer in plain text format - no meta-commentary or markdown formatting
"""


def get_create_document_conversation(content: str) -> list[dict[str, str]]:
    """
    Create a conversation prompt for generating a web document from source content. The outputs can be passed to inference_batch method to get a response.
    """

    conversation = [
        {"role": "system", "content": _CREATE_DOCUMENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _CREATE_DOCUMENT_USER_PROMPT.format(content=content),
        },
    ]

    return conversation


def get_rag_generation_conversation(
    context: str, question: str
) -> list[dict[str, str]]:
    conversation = [
        {"role": "system", "content": _RAG_GENERATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _RAG_GENERATION_USER_PROMPT.format(
                context=context, question=question
            ),
        },
    ]

    return conversation


def get_context_str_from_docs(
    docs: list[dict[str, str]], chars_per_doc: int = None, shuffle: bool = True
) -> str:
    """
    Convert a list of documents into a single context string for RAG generation.
    Uses neutral labels "Context n" (no URLs) and optionally shuffles to reduce position bias.
    """
    import random
    ordered = list(docs)
    if shuffle and len(ordered) > 1:
        random.shuffle(ordered)
    context_parts = []
    for i, doc in enumerate(ordered):
        content = doc.get("text")
        if content:
            if chars_per_doc is not None:
                content = content[:chars_per_doc]
            context_parts.append(f"Context {i + 1}:\n{content}")
    return "\n\n".join(context_parts)

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
