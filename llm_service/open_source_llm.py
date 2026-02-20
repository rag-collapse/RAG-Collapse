from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from .common_llm import CommonLLM
import gc


class OpenSourceLLM(CommonLLM):
    """
    Connects to a vLLM OpenAI-compatible server instead of loading
    the model in-process.  Start the server first (e.g. via the
    SLURM script ``scripts/start_llm_server.sh``).

    The ``api_base`` can be passed directly or read from the
    ``VLLM_API_BASE`` environment variable (e.g.
    ``http://gypsum-gpu188.unity.rc.umass.edu:5150/v1``).
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        api_base: str = None,
        api_key: str = "EMPTY",
        max_workers: int = 32,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_workers = max_workers

        api_base = api_base or os.environ.get("VLLM_API_BASE")
        if api_base is None:
            raise ValueError(
                "api_base must be provided or set via the VLLM_API_BASE "
                "environment variable (e.g. 'http://hostname:5150/v1')."
            )

        self.client = OpenAI(api_key=api_key, base_url=api_base)

        # Auto-discover the served model name from the running server.
        models = self.client.models.list()
        self.served_model_name = models.data[0].id

    def __repr__(self) -> str:
        return (
            f"OpenSourceLLM(model_name={self.model_name}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}, top_p={self.top_p}, "
            f"served_model={self.served_model_name})"
        )

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """
        Generate outputs for a batch of raw text prompts via the
        ``/v1/completions`` endpoint.  Requests are made concurrently
        so the server can batch them internally.
        """

        def _complete(prompt: str) -> str:
            response = self.client.completions.create(
                model=self.served_model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            return response.choices[0].text

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(_complete, prompts))

    def generate_single(self, prompt: str) -> str:
        """Generate output for a single raw-text prompt."""
        return self.generate_batch([prompt])[0]

    def inference_batch(self, conversations: list[list[dict[str, str]]]) -> list[str]:
        """
        Perform inference on a batch of conversations via the
        ``/v1/chat/completions`` endpoint.  Each conversation is a list
        of OpenAI-format message dicts.  The server applies the chat
        template automatically.
        """

        def _chat_complete(messages: list[dict[str, str]]) -> str:
            response = self.client.chat.completions.create(
                model=self.served_model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            return response.choices[0].message.content

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(_chat_complete, conversations))

class EmbeddingModel:
    def __init__(
        self, model_name: str, batch_size: int = 32, cache_dir: str = None
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)

    def __repr__(self) -> str:
        return (
            f"EmbeddingModel(model_name={self.model_name}, "
            f"batch_size={self.batch_size}, cache_dir={self.cache_dir})"
        )

    def embed_batch(self, texts: list[str], normalize: bool = True):
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )

    def embed_single(self, text: str, normalize: bool = True):
        return self.model.encode(text, normalize_embeddings=normalize)

    def similarity(self, embed1: np.ndarray, embed2: np.ndarray) -> float:
        return self.model.similarity(embed1, embed2)[0]

    def similarity_batch(self, embeds1: np.ndarray, embeds2: np.ndarray) -> np.ndarray:
        return self.model.similarity(embeds1, embeds2)

    def shutdown(self):
        del self.model
        gc.collect()
