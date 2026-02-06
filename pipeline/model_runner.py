import os
from typing import Any, Dict, List, Tuple

from llm_service.common_llm import CommonLLM
from llm_service.proprietary_llm import ProprietaryLLM
from llm_service.open_source_llm import OpenSourceLLM


def build_llm_from_env() -> Tuple[CommonLLM, str]:
    """
    Build an LLM instance based on MODEL_MODE.

    MODEL_MODE:
      - api   -> ProprietaryLLM (CPU-safe)
      - local -> OpenSourceLLM (GPU-only)

    MODEL_NAME controls which model is used.
    """
    model_mode = os.getenv("MODEL_MODE", "api").strip().lower()
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()

    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("MAX_TOKENS", "512"))
    top_p = float(os.getenv("TOP_P", "0.9"))

    if model_mode == "api":
        llm: CommonLLM = ProprietaryLLM(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return llm, model_name

    if model_mode == "local":
        if not os.getenv("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError(
                "MODEL_MODE=local requires a GPU, but CUDA_VISIBLE_DEVICES is empty."
            )

        max_model_len = int(os.getenv("MAX_MODEL_LEN", "8192"))
        gpu_memory_utilization = float(os.getenv("GPU_MEM_UTIL", "0.7"))

        llm = OpenSourceLLM(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_log_stats=True,
        )
        return llm, model_name

    raise ValueError("MODEL_MODE must be 'api' or 'local'")


def _normalize_generation(gen: Any) -> str:
    """Normalize outputs across backends."""
    if isinstance(gen, str):
        return gen
    if isinstance(gen, dict):
        return gen.get("response") or gen.get("text") or gen.get("content") or str(gen)
    return str(gen)


def sample_runs(
    llm: CommonLLM,
    conversation: List[Dict[str, str]],
    num_runs: int,
) -> List[str]:
    """
    Repeat the same conversation num_runs times, and run as a batch.
    Returns a list of plain-text answers.
    """
    batch = [conversation for _ in range(num_runs)]
    outputs = llm.inference_batch(batch)
    return [_normalize_generation(o) for o in outputs]
