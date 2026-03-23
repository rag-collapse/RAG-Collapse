from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
import os
from .common_llm import CommonLLM


class ServerLLM(CommonLLM):
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
        max_workers: int = 96,
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

        self.client = OpenAI(api_key=api_key, base_url=api_base, timeout=1200.0)

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

    def inference_agentic_single(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_executor: "Callable[[str, dict], str]",
        max_tool_calls: int = 10,
    ) -> "tuple[str, int]":
        """
        Single agentic loop: call the model, execute any tool calls it makes,
        append results, and repeat until finish_reason is 'stop' or max_tool_calls
        is exhausted (in which case one final call is made with tool_choice='none').

        tool_executor(name, args_dict) -> str
            Executes a named tool with parsed arguments and returns a plain-text result.

        Returns (answer, tool_calls_used) where tool_calls_used is the number of
        retrieve tool-call rounds executed before the final answer.

        Server must be started with:
            --enable-auto-tool-choice --tool-call-parser hermes   (Qwen2.5 family)
        For other models:
            --tool-call-parser mistral                             (Mistral / Mixtral)
            --tool-call-parser llama3_json                         (Llama-3.x)
            --enable-auto-tool-choice --tool-call-parser deepseek_v3 (DeepSeek-V3 / R1)
        """
        import json
        history = list(messages)
        tool_calls_used = 0

        for i in range(max_tool_calls):
            response = self.client.chat.completions.create(
                model=self.served_model_name,
                messages=history,
                tools=tools,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            choice = response.choices[0]

            if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
                # Model produced a final answer
                return choice.message.content or "", tool_calls_used

            tool_calls_used += 1

            # Append assistant message with tool_calls
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

            # Execute each tool call and append results
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
        response = self.client.chat.completions.create(
            model=self.served_model_name,
            messages=history,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            tool_choice="none",
        )
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
        Each conversation gets its own tool_executor (so per-question retrieval stores
        are isolated). Uses the same ThreadPoolExecutor pattern as inference_batch.

        conversations[i] and tool_executors[i] must correspond.

        Returns list of (answer, tool_calls_used) tuples.
        """
        def _run(args):
            messages, executor = args
            return self.inference_agentic_single(messages, tools, executor, max_tool_calls)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(_run, zip(conversations, tool_executors)))
