from typing import Any, Dict, List, Optional, Tuple

from llm_service.common_llm import CommonLLM
from llm_service.proprietary_llm import ProprietaryLLM
from llm_service.open_source_llm import OpenSourceLLM
from llm_service.server_llm import OpenSourceLLM as ServerLLM


def build_llm(
    *,
    model_mode: str,
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.9,
    # local-only knobs
    require_gpu: bool = True,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.7,
    cuda_visible_devices: Optional[str] = None,
    tensor_parallel_size: int = 1,
    # server-only knobs
    api_base: Optional[str] = None,
) -> Tuple[CommonLLM, str]:
    """
    Build and return an LLM instance from explicit arguments.

    model_mode:
      - "api"    -> ProprietaryLLM (CPU-safe, LiteLLM)
      - "local"  -> OpenSourceLLM (in-process vLLM, GPU-only)
      - "server" -> ServerLLM (HTTP client to a running vLLM server)

    Note:
      - API_KEY should be provided via environment variable for API mode
        (handled inside ProprietaryLLM / LiteLLM client).
      - cuda_visible_devices should be passed from the caller (SLURM sets it).
    """
    mode = model_mode.strip().lower()
    name = model_name.strip()

    if mode == "api":
        llm: CommonLLM = ProprietaryLLM(
            model_name=name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return llm, name

    if mode == "local":
        if require_gpu:
            if not (cuda_visible_devices and cuda_visible_devices.strip()):
                raise RuntimeError(
                    "model_mode=local requires a GPU, but cuda_visible_devices is empty."
                )

        llm = OpenSourceLLM(
            model_name=name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            disable_log_stats=True,
        )
        return llm, name

    if mode == "server":
        llm = ServerLLM(
            model_name=name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            api_base=api_base,
        )
        return llm, llm.served_model_name

    raise ValueError("model_mode must be 'api', 'local', or 'server'")


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
