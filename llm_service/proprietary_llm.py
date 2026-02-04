from litellm import completion, batch_completion #,_turn_on_debug
import os
from dotenv import load_dotenv
load_dotenv()

#_turn_on_debug() # Only turn on in case of debugging

class ProprietaryLLM:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.api_key = os.getenv("API_KEY")
        self.extra_params = kwargs

    def __repr__(self) -> str:
        return (
            f"ProprietaryLLM(model_name={self.model_name}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}, top_p={self.top_p})"
        )

    def _get_completion_params(self) -> dict:
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "api_base": "https://thekeymaker.umass.edu/",
            **self.extra_params,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def generate_single(self, messages: list[dict[str, str]]) -> str:
        params = self._get_completion_params()
        response = completion(messages=messages, **params)
        return response.choices[0].message.content

    def generate_batch(self, conversations: list[list[dict[str, str]]]) -> list[str]:
        params = self._get_completion_params()
        responses = batch_completion(messages=conversations, **params)
        results = []
        for response in responses:
            if isinstance(response, Exception):
                raise response
            results.append(response.choices[0].message.content)
        return results

    def inference_batch(self, conversations: list[list[dict[str, str]]]) -> list[str]:
        return self.generate_batch(conversations)