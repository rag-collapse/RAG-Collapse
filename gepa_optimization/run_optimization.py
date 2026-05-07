"""
GEPA optimization entry point.

Required environment variables:
  API_KEY          — UMass keymaker API key

Optional environment variables:
  LITELLM_API_BASE   — proxy URL                          (default: https://thekeymaker.umass.edu/)
  TASK_MODEL         — model for RAG runs                 (default: openai/claude-haiku-4-5)
  DOC_GEN_MODEL      — model for AI document generation   (default: openai/gemma-3-12b-it)
  JUDGE_MODEL        — model for quality                  (default: openai/gpt4o)
  REFLECTION_MODEL   — model for GEPA                     (default: openai/claude-opus-4-1)
  EMBED_MODEL        — local SentenceTransformer for search (default: all-MiniLM-L6-v2)
  MAX_METRIC_CALLS   — GEPA evaluation budget             (default: 100)

Usage (from repo root):
  python gepa_optimization/run_optimization.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import litellm
import gepa
from dotenv import load_dotenv

load_dotenv()

litellm.drop_params = True  # gpt5 and some models reject temperature=0; drop silently

from formatters import GEPA_RAG_GENERATION_SYSTEM_PROMPT
from rag_adapter import RAGSystemPromptAdapter
from prepare_dataset import RAGDataInst
from gepa_logger import GEPALogger, GEPALiteLLMCallback

# ── Shared config ─────────────────────────────────────────────────────────────
API_BASE = os.environ.get("LITELLM_API_BASE", "https://thekeymaker.umass.edu/")
API_KEY  = os.environ["API_KEY"]

TASK_MODEL       = os.environ.get("TASK_MODEL",       "openai/claude-haiku-4-5")
DOC_GEN_MODEL    = os.environ.get("DOC_GEN_MODEL",    "openai/gemma-3-12b-it")
JUDGE_MODEL      = os.environ.get("JUDGE_MODEL",      "openai/gpt4o")
REFLECTION_MODEL = os.environ.get("REFLECTION_MODEL", "openai/claude-opus-4-1")
EMBED_MODEL      = os.environ.get("EMBED_MODEL",      "all-MiniLM-L6-v2")
MAX_METRIC_CALLS = int(os.environ.get("MAX_METRIC_CALLS", "100"))

# ── Logger — captures every LLM call and structured evaluation events ─────────
RUN_DIR = "./gepa_runs/rag_system_prompt"
logger = GEPALogger(f"{RUN_DIR}/logs")
litellm.callbacks = [GEPALiteLLMCallback(logger)]

# ── Reflection LM — callable so proxy config doesn't pollute global litellm state ──
def reflection_lm(prompt: str | list[dict], **kwargs) -> str:
    # GEPA passes a plain string from prompt_renderer; wrap it into messages format
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt
    response = litellm.completion(
        model=REFLECTION_MODEL,
        messages=messages,
        api_base=API_BASE,
        api_key=API_KEY,
        temperature=1.0,
        max_tokens=2048,
        metadata={"role": "reflection"},
    )
    return response.choices[0].message.content

# ── Adapter ───────────────────────────────────────────────────────────────────
adapter = RAGSystemPromptAdapter(
    task_model=TASK_MODEL,
    doc_gen_model=DOC_GEN_MODEL,
    judge_model=JUDGE_MODEL,
    api_base=API_BASE,
    api_key=API_KEY,
    n_rounds=10,
    embed_model=EMBED_MODEL,
    chars_per_doc=800,
    logger=logger,
)

# ── Dataset ───────────────────────────────────────────────────────────────────
def _load_jsonl(path: Path) -> list[RAGDataInst]:
    with open(path, encoding="utf-8") as f:
        return [RAGDataInst(**json.loads(line)) for line in f if line.strip()]

data_dir = Path(__file__).parent / "data"
trainset = _load_jsonl(data_dir / "train.jsonl")
valset   = _load_jsonl(data_dir / "val.jsonl")

print(f"Dataset      — Train: {len(trainset)} | Val: {len(valset)}")
print(f"Task model   — {TASK_MODEL}")
print(f"Doc gen model— {DOC_GEN_MODEL}")
print(f"Judge model  — {JUDGE_MODEL}")
print(f"Reflection   — {REFLECTION_MODEL}")
print(f"Embed model  — {EMBED_MODEL}")
print(f"Proxy        — {API_BASE}")
print(f"Max calls    — {MAX_METRIC_CALLS}")
print(f"Variants     — replace_all, replace_one, search  (10 rounds each)")

# ── Optimize ──────────────────────────────────────────────────────────────────
result = gepa.optimize(
    seed_candidate={"system_prompt": GEPA_RAG_GENERATION_SYSTEM_PROMPT},
    trainset=trainset,
    valset=valset,
    adapter=adapter,
    reflection_lm=reflection_lm,
    max_metric_calls=MAX_METRIC_CALLS,
    candidate_selection_strategy="pareto",
    display_progress_bar=True,
    run_dir=RUN_DIR,
)

best_score = result.val_aggregate_scores[result.best_idx]

print("\n" + "=" * 60)
print("Best optimized system prompt:")
print("=" * 60)
print(result.best_candidate["system_prompt"])
print(f"\nBest val score (avg anti_collapse): {best_score:.4f}")
print(f"Pareto frontier: {len(result.per_val_instance_best_candidates)} val instances tracked")

print("\nTo apply: python gepa_optimization/apply_best_prompt.py --prompt '<prompt>'")
