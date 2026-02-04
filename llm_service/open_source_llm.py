from vllm import LLM, SamplingParams
import numpy as np
from transformers import AutoTokenizer
import warnings


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
        import gc

        gc.collect()


class EmbeddingModel:
    def __init__(self, model_name: str, cache_dir: str = None) -> None:
        raise NotImplementedError("This is a placeholder for an EmbeddingModel class.")

    # def embed_batch(self, texts: list[str]) -> np.ndarray | None:
    #     try:
    #         outputs = self.llm.embed(texts)

    #         embeddings = []
    #         for output in outputs:
    #             token_embeddings = output.token_embeddings
    #             avg_embedding = np.mean(token_embeddings, axis=0)
    #             embeddings.append(avg_embedding)

    #         return np.array(embeddings)
    #     except ValueError as e:
    #         warnings.warn(
    #             f"Embedding API is not supported by model '{self.model_name}'. "
    #             f"Error: {str(e)}. Try converting the model using `--convert embed` if needed. "
    #             f"Returning None.",
    #             UserWarning
    #         )
    #         return None

    # def embed_single(self, text: str) -> np.ndarray | None:
    #     result = self.embed_batch([text])
    #     return result[0] if result is not None else None
