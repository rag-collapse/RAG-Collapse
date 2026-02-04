from vllm import LLM, SamplingParams
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import warnings
import gc


class OpenSourceLLM:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        cache_dir: str = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.llm = LLM(model=model_name, download_dir=cache_dir, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p
        )

    def __repr__(self) -> str:
        return (
            f"OpenSourceLLM(model_name={self.model_name}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}, top_p={self.top_p}), "
            f"LLM: {self.llm}"
        )

    def apply_chat_template(
        self, messages: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Apply the model's chat template to OpenAI-format messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Example: [
                         {"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": "Hello!"}
                     ]
            add_generation_prompt: Whether to add the generation prompt (default: True).
                                  This typically adds the assistant's role prefix.

        Returns:
            Formatted prompt string ready for generation.
        """
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def apply_chat_template_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        add_generation_prompt: bool = True,
    ) -> list[str]:
        """
        Apply the model's chat template to a batch of OpenAI-format message lists.

        Args:
            messages_batch: List of message lists, where each message list contains
                           dicts with 'role' and 'content' keys.
            add_generation_prompt: Whether to add the generation prompt (default: True).

        Returns:
            List of formatted prompt strings ready for batch generation.
        """
        return [
            self.apply_chat_template(messages, add_generation_prompt)
            for messages in messages_batch
        ]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """
        Ensure chat templates are applied before calling this method. Generates outputs for a batch of prompts.

        Args:
            prompts: List of prompt strings.
        Returns:
            List of generated output strings corresponding to each prompt.
        """
        responses = self.llm.generate(
            prompts=prompts, sampling_params=self.sampling_params
        )
        return [response.outputs[0].text for response in responses]

    def generate_single(self, prompt: str) -> str:
        """Generates output for a single prompt."""
        return self.generate_batch([prompt])[0]

    def inference_batch(self, conversations: list[list[dict[str, str]]]) -> list[str]:
        """Performs inference on a batch of conversations. Only need to provide OpenAI-format messages.

        Args:
            conversations: List of conversations, where each conversation is a list of message dicts.
        Returns:
            List of generated output strings corresponding to each conversation.
        """
        templated_prompts = self.apply_chat_template_batch(
            conversations, add_generation_prompt=True
        )
        responses = self.generate_batch(templated_prompts)
        return responses

    def shutdown(self):
        """Properly shutdown the vLLM engine."""
        del self.llm
        gc.collect()


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
