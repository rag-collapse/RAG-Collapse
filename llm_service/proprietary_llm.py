import litellm
from litellm import completion, batch_completion #,_turn_on_debug
from litellm.exceptions import BadRequestError as LiteLLMBadRequestError
from litellm.exceptions import RateLimitError as LiteLLMRateLimitError
from .common_llm import CommonLLM
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
import os
import re
import time
import json
from dotenv import load_dotenv
load_dotenv()

# Reasoning models (e.g. gpt-5 / gpt-5-mini) reject sampling params like top_p and any
# non-default temperature. drop_params makes litellm silently strip params a model doesn't
# support instead of raising UnsupportedParamsError, so the same call works across model
# families. NOTE: these models also spend the token budget on hidden reasoning tokens — give
# them a generous max_tokens (>=~2000 for document generation) or the visible content is empty.
litellm.drop_params = True
# Keep litellm from echoing request/error info (which can include a key id) to stdout/stderr.
litellm.suppress_debug_info = True

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

    def _redact(self, msg: object) -> str:
        """Strip secrets/ids from an error message before it can reach a log or traceback:
        the configured API key, any long hex run (provider key-id / hash), and sk-style keys."""
        s = str(msg)
        if self.api_key:
            s = s.replace(self.api_key, "***REDACTED***")
        s = re.sub(r"[A-Fa-f0-9]{32,}", "***", s)            # hashed key-ids (e.g. keymaker/Azure)
        s = re.sub(r"sk-[A-Za-z0-9_\-]{12,}", "***", s)        # raw OpenAI-style keys
        return s

    def generate_single(self, messages: list[dict[str, str]]) -> str:
        params = self._get_completion_params()
        try:
            response = completion(messages=messages, **params)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"ProprietaryLLM completion failed: {self._redact(e)}") from None

    def generate_batch(self, conversations: list[list[dict[str, str]]]) -> list[str]:
        """Batched completion that stays under provider token-rate limits.

        Large concurrent batches (e.g. distractor synthesis over many questions) can exceed the
        keymaker/Azure tokens-per-minute limit and raise RateLimitError. We send the batch in
        chunks (bounding concurrency/burst) and retry a rate-limited chunk with backoff so a
        per-minute limit becomes a brief wait instead of a crash. Tunable via env:
          LITELLM_BATCH_CHUNK (default 12), LITELLM_BATCH_PACING_SEC (3), LITELLM_RATE_RETRIES (6).
        """
        params = self._get_completion_params()
        chunk = max(1, int(os.getenv("LITELLM_BATCH_CHUNK", "12")))
        pace = float(os.getenv("LITELLM_BATCH_PACING_SEC", "3"))
        max_attempts = max(1, int(os.getenv("LITELLM_RATE_RETRIES", "6")))
        results: list[str] = []
        n = len(conversations)
        for start in range(0, n, chunk):
            sub = conversations[start:start + chunk]
            for attempt in range(max_attempts):
                try:
                    responses = batch_completion(messages=sub, **params)
                    bad = next((r for r in responses if isinstance(r, Exception)), None)
                    if bad is not None:
                        raise bad
                    results.extend(r.choices[0].message.content for r in responses)
                    break
                except LiteLLMRateLimitError as e:
                    if attempt == max_attempts - 1:
                        raise RuntimeError(
                            f"rate limit not cleared after {max_attempts} retries: {self._redact(e)}") from None
                    wait = min(90, 20 * (attempt + 1))   # ~20s,40s,60s,80s,90s — spans a 1-min reset
                    print(f"[ProprietaryLLM] rate-limited; backoff {wait}s "
                          f"(chunk {start}-{start+len(sub)}, attempt {attempt+1}/{max_attempts})", flush=True)
                    time.sleep(wait)
                except Exception as e:
                    raise RuntimeError(f"ProprietaryLLM batch failed: {self._redact(e)}") from None
            if pace > 0 and start + chunk < n:
                time.sleep(pace)
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