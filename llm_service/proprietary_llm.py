from litellm import completion, batch_completion #,_turn_on_debug
from litellm.exceptions import BadRequestError as LiteLLMBadRequestError
from .common_llm import CommonLLM
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
import os
import json
from dotenv import load_dotenv
load_dotenv()

#_turn_on_debug() # Only turn on in case of debugging

class ProprietaryLLM(CommonLLM):
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

    def inference_agentic_single(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_executor: "Callable[[str, dict], str]",
        max_tool_calls: int = 10,
    ) -> "tuple[str, int]":
        """
        Single agentic loop using litellm (OpenAI-compatible API).
        Mirrors ServerLLM.inference_agentic_single — same message format,
        same tool_choice escalation strategy.

        Returns (answer, tool_calls_used).
        """
        history = list(messages)
        tool_calls_used = 0
        params = self._get_completion_params()

        for i in range(max_tool_calls):
            # First call: require at least one tool use; after that let the model decide.
            tc_mode = "required" if i == 0 else "auto"
            try:
                response = completion(
                    messages=history,
                    tools=tools,
                    tool_choice=tc_mode,
                    **params,
                )
            except LiteLLMBadRequestError:
                if tc_mode != "required":
                    raise
                # Upstream proxy (e.g. keymaker → Azure) doesn't support tool_choice="required".
                # Fall back to "auto" for this call and all subsequent ones.
                tc_mode = "auto"
                response = completion(
                    messages=history,
                    tools=tools,
                    tool_choice="auto",
                    **params,
                )
            choice = response.choices[0]

            if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
                return choice.message.content or "", tool_calls_used

            tool_calls_used += 1

            history.append({
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ],
            })

            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result = tool_executor(tc.function.name, args)
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # max_tool_calls exhausted — force a final text answer
        response = completion(messages=history, tool_choice="none", **params)
        return response.choices[0].message.content or "", tool_calls_used

    def inference_agentic_batch(
        self,
        conversations: list[list[dict]],
        tools: list[dict],
        tool_executors: "list[Callable[[str, dict], str]]",
        max_tool_calls: int = 10,
    ) -> "list[tuple[str, int]]":
        """
        Parallel agentic inference across a batch of conversations.
        Returns list of (answer, tool_calls_used) tuples.
        """
        if len(conversations) != len(tool_executors):
            raise ValueError(
                f"conversations and tool_executors must have the same length, "
                f"got {len(conversations)} and {len(tool_executors)}"
            )

        def _run(args):
            messages, executor = args
            return self.inference_agentic_single(messages, tools, executor, max_tool_calls)

        with ThreadPoolExecutor() as executor:
            return list(executor.map(_run, zip(conversations, tool_executors)))