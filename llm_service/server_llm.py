from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
import os
import threading
import time
from .common_llm import CommonLLM


def _strip_reasoning(text: "str | None") -> "str | None":
    """Remove a DeepSeek-R1-style ``<think>...</think>`` reasoning block from a response.

    Reasoning models (e.g. DeepSeek-R1-Distill) are served here WITHOUT vLLM's
    ``--reasoning-parser`` because that parser corrupts ``message.content`` into
    byte-level BPE artifacts (``Ġ``/``Ċ``) on current vLLM. Served without it, the model
    emits ``<think> ... </think>`` followed by the final answer as ordinary, cleanly
    decoded text — so we keep only the text after the last ``</think>``.

    No-op for non-reasoning models (no ``</think>`` present) and for content already
    cleaned by a working reasoning parser.
    """
    if not text:
        return text
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.lstrip()


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
        enable_thinking: bool = True,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_workers = max_workers
        # extra_body is passed to every chat completion request.
        # Set enable_thinking=False for Qwen3/Qwen3.5 non-reasoning mode.
        # Requires the server to be started with --reasoning-parser qwen3.
        self._extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if not enable_thinking else {}

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
            return _strip_reasoning(response.choices[0].text)

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
                extra_body=self._extra_body or None,
            )
            return _strip_reasoning(response.choices[0].message.content)

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

        def _create_with_retry(**kwargs):
            for attempt in range(3):
                try:
                    return self.client.chat.completions.create(**kwargs)
                except (
                    InternalServerError,   # 500 — vLLM transient crash
                    RateLimitError,        # 429 — server backpressure
                    APIConnectionError,    # network blip
                    APITimeoutError,       # request timed out
                ):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)

        history = list(messages)
        tool_calls_used = 0

        for i in range(max_tool_calls):
            # First call: require at least one tool use; after that let the model decide.
            tc_mode = "required" if i == 0 else "auto"
            try:
                response = _create_with_retry(
                    model=self.served_model_name,
                    messages=history,
                    tools=tools,
                    tool_choice=tc_mode,
                    parallel_tool_calls=False,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    extra_body=self._extra_body or None,
                )
            except BadRequestError:
                if tc_mode != "required":
                    raise
                # Model/parser doesn't support tool_choice="required" (e.g. llama3_json).
                # Fall back to "auto" for this and all subsequent calls.
                response = _create_with_retry(
                    model=self.served_model_name,
                    messages=history,
                    tools=tools,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    extra_body=self._extra_body or None,
                )
            choice = response.choices[0]

            if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
                # Model produced a final answer
                return _strip_reasoning(choice.message.content or ""), tool_calls_used

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
        response = _create_with_retry(
            model=self.served_model_name,
            messages=history,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            tool_choice="none",
        )
        return _strip_reasoning(response.choices[0].message.content or ""), tool_calls_used

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
        if len(conversations) != len(tool_executors):
            raise ValueError(
                f"conversations and tool_executors must have the same length, "
                f"got {len(conversations)} and {len(tool_executors)}"
            )

        sem = threading.Semaphore(32)

        def _run(args):
            messages, executor = args
            with sem:
                return self.inference_agentic_single(messages, tools, executor, max_tool_calls)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(_run, zip(conversations, tool_executors)))
