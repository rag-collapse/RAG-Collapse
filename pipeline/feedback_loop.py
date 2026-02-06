from typing import Dict, List, Any

def references_to_documents(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert example['references'] into a list of 'documents'.
    Assumes each reference has {url, text}.
    """
    docs = []
    for r in example.get("references", []):
        docs.append(
            {
                "url": r.get("url", ""),
                "text": r.get("text", ""),
            }
        )
    return docs

def answers_to_documents(answers: List[str]) -> List[Dict[str, str]]:
    """
    Wrap model answers as 'documents' so they can be injected/retrieved later.
    """
    return [{"url": "model_generated", "text": a} for a in answers]
