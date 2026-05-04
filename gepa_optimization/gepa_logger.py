"""
GEPA optimization logger — writes JSON Lines to a log directory.

Every LLM call (judge, reflection, doc-gen, task) is intercepted via
litellm's CustomLogger and written to llm_calls.jsonl with its role,
full messages, full response, token usage, latency, and any error.

Structured events (prompt candidates, per-variant evaluations, aggregates)
are written to separate .jsonl files for easy analysis.

Usage
-----
    from gepa_logger import GEPALogger, GEPALiteLLMCallback
    import litellm

    logger = GEPALogger("./gepa_runs/rag_system_prompt/logs")
    litellm.callbacks = [GEPALiteLLMCallback(logger)]
"""

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import litellm
from litellm.integrations.custom_logger import CustomLogger


# ── Utilities ─────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def candidate_id_from_prompt(prompt_text: str) -> str:
    """Stable 8-char ID derived from prompt content (same text → same ID)."""
    return "c_" + hashlib.sha256(prompt_text.encode()).hexdigest()[:8]


# ── Core logger ───────────────────────────────────────────────────────────────

class GEPALogger:
    """
    Thread-safe JSON Lines logger for a single GEPA run.

    Files written under log_dir:
      run_meta.jsonl      — run start metadata (one record)
      prompts.jsonl       — every prompt candidate introduced (seed or mutation)
      evaluations.jsonl   — per-question, per-variant scoring detail
      aggregates.jsonl    — per-question averaged scores across all variants
      llm_calls.jsonl     — every litellm call (via GEPALiteLLMCallback)
    """

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._write("run_meta", {
            "run_id": uuid.uuid4().hex[:8],
            "log_dir": str(self.log_dir),
        })

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write(self, event_type: str, record: dict):
        record["_type"] = event_type
        record["_ts"] = _now()
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            with (self.log_dir / f"{event_type}.jsonl").open("a", encoding="utf-8") as f:
                f.write(line)

    # ── Public API ────────────────────────────────────────────────────────────

    def log_prompt(
        self,
        candidate_id: str,
        iteration: int,
        prompt_text: str,
        source: str = "mutation",
    ):
        """Log a prompt candidate being introduced (seed on iter 0, mutation after)."""
        self._write("prompts", {
            "candidate_id": candidate_id,
            "iteration": iteration,
            "source": source,
            "prompt_text": prompt_text,
        })

    def log_evaluation(
        self,
        candidate_id: str,
        iteration: int,
        question: str,
        variant: str,
        round_answers: list[str],
        anti_collapse: float,
        unique_entities: int,
        quality: float,
    ):
        """Log per-question, per-variant scoring detail including every round answer."""
        self._write("evaluations", {
            "candidate_id": candidate_id,
            "iteration": iteration,
            "question": question,
            "variant": variant,
            "round_answers": round_answers,
            "anti_collapse": round(anti_collapse, 4),
            "unique_entities": unique_entities,
            "quality": round(quality, 4),
        })

    def log_aggregate(
        self,
        candidate_id: str,
        iteration: int,
        question: str,
        avg_anti_collapse: float,
        avg_quality: float,
        per_variant: dict,
    ):
        """Log averaged scores across all variants for a question."""
        self._write("aggregates", {
            "candidate_id": candidate_id,
            "iteration": iteration,
            "question": question,
            "avg_anti_collapse": round(avg_anti_collapse, 4),
            "avg_quality": round(avg_quality, 4),
            "per_variant": per_variant,
        })


# ── litellm callback ──────────────────────────────────────────────────────────

class GEPALiteLLMCallback(CustomLogger):
    """
    Intercepts every litellm.completion call and writes it to llm_calls.jsonl.

    The metadata dict passed to litellm.completion() is forwarded here and
    used to tag each record with:
      role         — entity_extraction | entity_clustering | quality_judge |
                     reflection | doc_gen | task | unknown
      candidate_id — which prompt candidate is being evaluated
      question     — the RAG question being processed
      variant      — collapse variant (replace_all / replace_one / search)
      iteration    — GEPA iteration number

    Captures on both success and failure so no call is ever silently dropped.
    """

    def __init__(self, logger: GEPALogger):
        super().__init__()
        self._logger = logger

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj, start_time, end_time, error=None)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        error = str(kwargs.get("exception", "unknown"))
        self._record(kwargs, response_obj, start_time, end_time, error=error)

    def _record(self, kwargs, response_obj, start_time, end_time, error):
        litellm_params = kwargs.get("litellm_params") or {}
        metadata = litellm_params.get("metadata") or {}

        try:
            response_text = response_obj.choices[0].message.content if response_obj else None
        except Exception:
            response_text = None

        try:
            usage = {
                "prompt_tokens": response_obj.usage.prompt_tokens,
                "completion_tokens": response_obj.usage.completion_tokens,
                "total_tokens": response_obj.usage.total_tokens,
            } if (response_obj and response_obj.usage) else {}
        except Exception:
            usage = {}

        try:
            duration_s = round((end_time - start_time).total_seconds(), 3)
        except Exception:
            duration_s = None

        self._logger._write("llm_calls", {
            "model": kwargs.get("model"),
            "role": metadata.get("role", "unknown"),
            "candidate_id": metadata.get("candidate_id"),
            "question": metadata.get("question"),
            "variant": metadata.get("variant"),
            "iteration": metadata.get("iteration"),
            "messages": kwargs.get("messages"),
            "response_text": response_text,
            "usage": usage,
            "duration_s": duration_s,
            "error": error,
        })
