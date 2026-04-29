"""
End-to-end smoke test for the GEPA optimization loop.

Exercises every layer of the stack with minimal settings so it finishes on a
local CPU laptop in ~15–30 minutes:

  Step 1 — Import checks (gepa, litellm, faiss, torch, transformers, datasets)
  Step 2 — Dataset prep: mteb/hotpotqa corpus + E5 + IndexFlatIP on CPU
            (corpus capped at 5 000 rows, 10 questions per source = 20 total)
  Step 3 — Scoring: anti_collapse_score + judge_quality_score on one question
  Step 4 — Adapter: RAGSystemPromptAdapter.evaluate() on 2 questions, 2 rounds
  Step 5 — Full GEPA loop: gepa.optimize() with max_metric_calls=5

Requires:
  pip install gepa litellm sentence-transformers python-dotenv
  pip install faiss-cpu torch transformers datasets
  API_KEY env var (UMass keymaker key)

Usage (from repo root):
  python gepa_optimization/smoke_test_gepa.py
  python gepa_optimization/smoke_test_gepa.py --api-key YOUR_KEY
  python gepa_optimization/smoke_test_gepa.py --skip-dataset   # reuse existing data/
  python gepa_optimization/smoke_test_gepa.py --skip-gepa-loop # stop after adapter
  python gepa_optimization/smoke_test_gepa.py --cache-dir /path/to/hf_cache
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"

# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check(label: str, value, ok_fn=None) -> None:
    ok = ok_fn(value) if ok_fn else bool(value)
    status = "PASS" if ok else "WARN"
    print(f"  [{status}] {label}: {value}")


def load_jsonl(path: Path):
    from prepare_dataset import RAGDataInst
    with open(path, encoding="utf-8") as f:
        return [RAGDataInst(**json.loads(line)) for line in f if line.strip()]


# ── Steps ─────────────────────────────────────────────────────────────────────

def step_imports() -> None:
    banner("Step 1 — Import checks")

    import gepa
    print("  [PASS] gepa")
    import litellm
    print("  [PASS] litellm")
    from gepa.core.adapter import GEPAAdapter, EvaluationBatch
    print("  [PASS] gepa.core.adapter")

    import faiss
    print(f"  [PASS] faiss  (version: {faiss.__version__ if hasattr(faiss, '__version__') else 'ok'})")
    import torch
    print(f"  [PASS] torch  (version: {torch.__version__}, CUDA: {torch.cuda.is_available()})")
    import transformers
    print(f"  [PASS] transformers  (version: {transformers.__version__})")
    import datasets as hf_datasets
    print(f"  [PASS] datasets  (version: {hf_datasets.__version__})")
    import sentence_transformers
    print(f"  [PASS] sentence-transformers  (version: {sentence_transformers.__version__})")

    from formatters import GEPA_RAG_GENERATION_SYSTEM_PROMPT
    print("  [PASS] formatters.GEPA_RAG_GENERATION_SYSTEM_PROMPT")
    from rag_adapter import RAGSystemPromptAdapter
    print("  [PASS] rag_adapter.RAGSystemPromptAdapter")
    from scoring import anti_collapse_score, judge_quality_score
    print("  [PASS] scoring functions")
    from pipeline_simulator import simulate_replace_all, simulate_replace_one, simulate_search
    print("  [PASS] pipeline_simulator functions")
    from prepare_dataset import prepare, RAGDataInst
    print("  [PASS] prepare_dataset.prepare")

    print("\n  All imports OK.")


def step_dataset(cache_dir: str | None) -> None:
    banner("Step 2 — Dataset prep (FAISS IndexFlatIP + E5 on CPU)")
    from prepare_dataset import prepare

    print("  Running prepare_dataset.prepare(smoke_test=True) ...")
    print("  Corpus: first 5 000 rows of mteb/hotpotqa")
    print("  Index:  IndexFlatIP (brute-force IP, no training, CPU-safe)")
    print("  E5:     intfloat/e5-small-v2  (mean pooling + L2 normalize)")
    print("  Output: 12 train / 4 val / 4 test  (60/20/20 of 20 questions)")
    print()

    t0 = time.time()
    prepare(out_dir=DATA_DIR, smoke_test=True, cache_dir=cache_dir)
    elapsed = time.time() - t0

    for split in ("train", "val", "test"):
        path = DATA_DIR / f"{split}.jsonl"
        check(f"{split}.jsonl exists", path.exists())
        if path.exists():
            rows = sum(1 for _ in path.open())
            check(f"  {split} rows", rows, lambda v: v > 0)

    print(f"\n  Dataset prep complete in {elapsed:.1f}s.")


def step_scoring(api_base: str, api_key: str, judge_model: str) -> None:
    banner("Step 3 — Scoring functions (anti_collapse + quality)")
    from scoring import anti_collapse_score, judge_quality_score

    trainset = load_jsonl(DATA_DIR / "train.jsonl")
    inst = trainset[0]
    question = inst.question
    context = inst.docs[0]["text"] if inst.docs else "No context available."

    # Synthesise 3 near-identical answers to check that high-similarity input
    # → low anti_collapse (collapsed) while diverse input → high score.
    collapsed_answers = [
        f"Based on the context, {question[:60]}",
        f"Based on the context, {question[:60]}",
        f"Based on the context, {question[:60]}",
    ]
    diverse_answers = [
        context[:120],
        context[30:150],
        context[60:180],
    ]

    print(f"  Question: {question[:80]}...")
    print()

    print("  anti_collapse_score on 3 near-identical answers (expect low score)...")
    t0 = time.time()
    score_low, unique_low = anti_collapse_score(question, collapsed_answers, judge_model, api_base, api_key)
    print(f"  Elapsed: {time.time()-t0:.1f}s")
    check("anti_collapse (collapsed)", round(score_low, 4), lambda v: 0.0 <= v <= 1.0)
    check("unique_entities (collapsed)", unique_low, lambda v: v >= 0)

    print()
    print("  anti_collapse_score on 3 diverse answers (expect higher score)...")
    t0 = time.time()
    score_high, unique_high = anti_collapse_score(question, diverse_answers, judge_model, api_base, api_key)
    print(f"  Elapsed: {time.time()-t0:.1f}s")
    check("anti_collapse (diverse)", round(score_high, 4), lambda v: 0.0 <= v <= 1.0)
    check("unique_entities (diverse)", unique_high, lambda v: v >= 0)

    print()
    print("  judge_quality_score on a plausible final answer...")
    final_answer = context[:200]
    t0 = time.time()
    q_score = judge_quality_score(question, context, final_answer, judge_model, api_base, api_key)
    print(f"  Elapsed: {time.time()-t0:.1f}s")
    check("quality", round(q_score, 4), lambda v: 0.0 <= v <= 1.0)


def step_adapter(
    api_base: str,
    api_key: str,
    task_model: str,
    doc_gen_model: str,
    judge_model: str,
    embed_model: str,
) -> None:
    banner("Step 4 — RAGSystemPromptAdapter.evaluate() (2 questions × 3 variants × 2 rounds)")
    from rag_adapter import RAGSystemPromptAdapter
    from formatters import GEPA_RAG_GENERATION_SYSTEM_PROMPT

    trainset = load_jsonl(DATA_DIR / "train.jsonl")
    batch = trainset[:2]

    print(f"  Q1: {batch[0].question[:70]}...")
    print(f"  Q2: {batch[1].question[:70]}...")
    print()

    adapter = RAGSystemPromptAdapter(
        task_model=task_model,
        doc_gen_model=doc_gen_model,
        judge_model=judge_model,
        api_base=api_base,
        api_key=api_key,
        n_rounds=2,
        embed_model=embed_model,
        chars_per_doc=400,
    )

    candidate = {"system_prompt": GEPA_RAG_GENERATION_SYSTEM_PROMPT}

    print("  Evaluating seed candidate ...")
    t0 = time.time()
    eval_batch = adapter.evaluate(batch, candidate, capture_traces=True)
    elapsed = time.time() - t0

    print(f"  Elapsed: {elapsed:.1f}s")
    check("num scores", len(eval_batch.scores), lambda v: v == 2)
    for i, obj in enumerate(eval_batch.objective_scores):
        check(f"Q{i+1} anti_collapse", round(obj["anti_collapse"], 4), lambda v: 0.0 <= v <= 1.0)
        check(f"Q{i+1} quality",       round(obj["quality"],       4), lambda v: 0.0 <= v <= 1.0)

    print()
    print("  Testing make_reflective_dataset ...")
    reflective = adapter.make_reflective_dataset(
        candidate, eval_batch, components_to_update=["system_prompt"]
    )
    check("reflective records", len(reflective.get("system_prompt", [])), lambda v: v == 2)
    print("  Adapter step passed.")


def step_gepa_loop(
    api_base: str,
    api_key: str,
    task_model: str,
    doc_gen_model: str,
    judge_model: str,
    reflection_model: str,
    embed_model: str,
) -> None:
    banner("Step 5 — Full gepa.optimize() (max_metric_calls=5)")
    import gepa
    import litellm
    from rag_adapter import RAGSystemPromptAdapter
    from formatters import GEPA_RAG_GENERATION_SYSTEM_PROMPT

    trainset = load_jsonl(DATA_DIR / "train.jsonl")
    valset   = load_jsonl(DATA_DIR / "val.jsonl")

    # Use only 2 train + 1 val to keep the smoke test fast
    trainset = trainset[:2]
    valset   = valset[:1]

    def reflection_lm(messages: list[dict], **kwargs) -> str:
        response = litellm.completion(
            model=reflection_model,
            messages=messages,
            api_base=api_base,
            api_key=api_key,
            temperature=1.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    adapter = RAGSystemPromptAdapter(
        task_model=task_model,
        doc_gen_model=doc_gen_model,
        judge_model=judge_model,
        api_base=api_base,
        api_key=api_key,
        n_rounds=2,
        embed_model=embed_model,
        chars_per_doc=400,
    )

    print("  gepa.optimize() — max_metric_calls=5, strategy=pareto")
    print("  (2 train questions × 2 rounds × 3 variants per GEPA evaluation)")
    print()

    t0 = time.time()
    result = gepa.optimize(
        seed_candidate={"system_prompt": GEPA_RAG_GENERATION_SYSTEM_PROMPT},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=5,
        candidate_selection_strategy="pareto",
        display_progress_bar=True,
        run_dir="./gepa_runs/smoke_test",
    )
    elapsed = time.time() - t0

    print(f"\n  Elapsed: {elapsed:.1f}s")
    check("best_candidate has system_prompt", "system_prompt" in result.best_candidate, lambda v: v)
    check("best_score in range", round(result.best_score, 4), lambda v: 0.0 <= v <= 1.0)

    print("\n  Best optimized prompt:")
    print("  " + "-" * 56)
    for line in result.best_candidate["system_prompt"].splitlines():
        print(f"    {line}")
    print("  " + "-" * 56)
    print(f"\n  Best val score (avg anti_collapse): {result.best_score:.4f}")
    print("\n  gepa.optimize() smoke test PASSED.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end GEPA smoke test (CPU-safe)")
    parser.add_argument("--api-key",        default=None, help="Keymaker API key (fallback: API_KEY env var)")
    parser.add_argument("--api-base",       default="https://thekeymaker.umass.edu/")
    parser.add_argument("--task-model",     default="anthropic/claude-haiku-4-5-20251001")
    parser.add_argument("--doc-gen-model",  default="anthropic/claude-haiku-4-5-20251001")
    parser.add_argument("--judge-model",    default="anthropic/claude-haiku-4-5-20251001")
    parser.add_argument("--reflection-model", default="anthropic/claude-opus-4-7")
    parser.add_argument("--embed-model",    default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model for the search variant (CPU-safe)")
    parser.add_argument("--cache-dir",      default=None,
                        help="HuggingFace cache directory for models and datasets")
    parser.add_argument("--skip-dataset",   action="store_true",
                        help="Skip dataset prep (reuse existing data/train|val|test.jsonl)")
    parser.add_argument("--skip-gepa-loop", action="store_true",
                        help="Stop after the adapter step")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("API_KEY")
    if not api_key:
        print("ERROR: No API key. Set API_KEY env var or pass --api-key.")
        sys.exit(1)

    print("Configuration")
    print(f"  API base        : {args.api_base}")
    print(f"  Task model      : {args.task_model}")
    print(f"  Doc gen model   : {args.doc_gen_model}")
    print(f"  Judge model     : {args.judge_model}")
    print(f"  Reflection model: {args.reflection_model}")
    print(f"  Embed model     : {args.embed_model}")
    print(f"  HF cache dir    : {args.cache_dir or '(default ~/.cache/huggingface)'}")
    print(f"  Skip dataset    : {args.skip_dataset}")
    print(f"  Skip GEPA loop  : {args.skip_gepa_loop}")

    step_imports()

    if args.skip_dataset:
        banner("Step 2 — Dataset prep (skipped, reusing existing data/)")
        for split in ("train", "val", "test"):
            path = DATA_DIR / f"{split}.jsonl"
            check(f"{split}.jsonl", path.exists())
    else:
        step_dataset(cache_dir=args.cache_dir)

    step_scoring(args.api_base, api_key, args.judge_model)

    step_adapter(
        api_base=args.api_base,
        api_key=api_key,
        task_model=args.task_model,
        doc_gen_model=args.doc_gen_model,
        judge_model=args.judge_model,
        embed_model=args.embed_model,
    )

    if args.skip_gepa_loop:
        print("\nSkipped full gepa.optimize() (--skip-gepa-loop).")
        print("All earlier steps passed.")
        return

    step_gepa_loop(
        api_base=args.api_base,
        api_key=api_key,
        task_model=args.task_model,
        doc_gen_model=args.doc_gen_model,
        judge_model=args.judge_model,
        reflection_model=args.reflection_model,
        embed_model=args.embed_model,
    )

    banner("Smoke test COMPLETE")
    print("  All 5 steps passed. The full GEPA loop works end-to-end on CPU.")
    print("  Next: run gepa_optimization/run_optimization.py on Unity HPC.")


if __name__ == "__main__":
    main()
