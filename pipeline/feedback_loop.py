from typing import Dict, List


def inject_generated_text(
    example: Dict,
    generated_text: str,
) -> Dict:
    """
    Create a new example where model-generated text
    is injected back as context.

    Original references are preserved separately.
    """
    return {
        "question": example["question"],
        "references": [
            {
                "url": "model_generated",
                "text": generated_text,
            }
        ],
        "original_references": example["references"],
    }
