"""
Retrieval for the Search pipeline variant: chunk documents, embed via LiteLLM, retrieve top-k.
"""

from typing import Any, Callable, Dict, List

from pipeline.config import SEARCH_CHUNK_OVERLAP, SEARCH_CHUNK_SIZE, SEARCH_TOP_K


def chunk_text(
    text: str,
    chunk_size: int = SEARCH_CHUNK_SIZE,
    overlap: int = SEARCH_CHUNK_OVERLAP,
) -> List[str]:
    """Split text into overlapping chunks (character-based)."""
    if not text or chunk_size <= 0:
        return [text] if text else []
    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start += step
    return [c for c in chunks if c]


def chunks_from_docs(
    docs: List[Dict[str, Any]],
    chunk_size: int = SEARCH_CHUNK_SIZE,
    overlap: int = SEARCH_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Turn each doc's text into chunks; each chunk keeps doc_id and iteration for tracing."""
    out = []
    for doc in docs:
        text = doc.get("text") or ""
        for i, ct in enumerate(chunk_text(text, chunk_size, overlap)):
            out.append({
                "doc_id": doc.get("doc_id", ""),
                "iteration": doc.get("iteration", -1),
                "chunk_index": i,
                "text": ct,
            })
    return out


class ChunkedRetrievalStore:
    """
    In-memory store of chunk embeddings. Add documents (as doc dicts with 'text'),
    then search by query string for top-k chunks. Returns list of doc-like dicts with 'text'.
    """

    def __init__(
        self,
        embed_fn: Callable[[List[str]], Any],
        chunk_size: int = SEARCH_CHUNK_SIZE,
        chunk_overlap: int = SEARCH_CHUNK_OVERLAP,
    ):
        self.embed_fn = embed_fn
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: List[Dict[str, Any]] = []
        self._vectors: Any = None  # numpy array

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Chunk docs and append to store; re-embed all for simplicity."""
        import numpy as np
        new_chunks = chunks_from_docs(docs, self.chunk_size, self.chunk_overlap)
        if not new_chunks:
            return
        texts = [c["text"] for c in new_chunks]
        arr = _to_float32(self.embed_fn(texts))
        if self._vectors is None:
            self._vectors = arr
        else:
            self._vectors = np.vstack([self._vectors, arr])
        self._chunks.extend(new_chunks)

    def search(self, query: str, k: int = SEARCH_TOP_K) -> List[Dict[str, Any]]:
        """Return top-k chunks by similarity to query. Returns doc-like dicts with 'text' for prompt."""
        import numpy as np
        if not self._chunks:
            return []
        qvec = _to_float32(self.embed_fn([query])).reshape(1, -1)
        # Normalize for cosine similarity
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        v_norm = self._vectors / norms
        q_norm = qvec / (np.linalg.norm(qvec, axis=1, keepdims=True) or 1)
        scores = np.dot(v_norm, q_norm.T).flatten()
        top_indices = np.argsort(scores)[::-1][:k]
        return [self._chunks[i] for i in top_indices]

    def __len__(self) -> int:
        return len(self._chunks)


# Same api_base as completion (ProprietaryLLM) so one config story: API_KEY only
_LITELLM_API_BASE = "https://thekeymaker.umass.edu/"


def _to_float32(vectors: Any) -> Any:
    """Ensure embeddings are a float32 numpy array (shared by local and API embed fns)."""
    import numpy as np
    return np.asarray(vectors, dtype=np.float32)


def _default_embed_cache_dir() -> str:
    """Single place for embed model cache (HF_HOME or ~/.cache/hf)."""
    import os
    return os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/hf")


def make_embed_fn_local(
    model_name: str = "all-MiniLM-L6-v2",
    cache_dir: str = None,
):
    """
    Return an embed function using local SentenceTransformer (EmbeddingModel).
    Use when the API does not expose an embedding model. cache_dir defaults to HF_HOME or ~/.cache/hf.
    """
    from llm_service.open_source_llm import EmbeddingModel
    if cache_dir is None:
        cache_dir = _default_embed_cache_dir()
    _embedding_model = EmbeddingModel(model_name=model_name, cache_dir=cache_dir)

    def embed(texts: List[str]):
        return _to_float32(_embedding_model.embed_batch(texts, normalize=True))

    return embed


def make_embed_fn_litellm(model: str = "text-embedding-ada-002"):
    """
    Return an embed function using LiteLLM's embedding().
    Uses the same config as completion: API_KEY from env, same api_base.
    """
    import os
    from litellm import embedding

    api_key = os.environ.get("API_KEY", "")

    def embed(texts: List[str]):
        kwargs = {
            "model": model,
            "input": texts,
            "api_base": _LITELLM_API_BASE,
        }
        if api_key:
            kwargs["api_key"] = api_key
        response = embedding(**kwargs)
        data = response.get("data", [])
        ordered = sorted(data, key=lambda x: x.get("index", 0))
        vectors = [d["embedding"] for d in ordered]
        return _to_float32(vectors)

    return embed
