import argparse
import os
from typing import Any, Dict, List

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    def record(fn):
        return fn  # no-op when not running under torch.distributed.run

from pipeline.config import (
    PIPELINE_VARIANTS,
    get_rounds_for_variant,
    is_hybrid,
    is_replace_one,
    is_search,
    SEARCH_TOP_K,
)
from pipeline.context_builder import (
    HybridContextConfig,
    get_initial_documents_hybrid,
    get_initial_documents_replace_one,
    get_next_documents,
)
from pipeline.data_loader import load_dataset, prepare_dataset
from pipeline.prompt_builder import build_rag_conversation
from pipeline.model_runner import build_llm
from pipeline.feedback_loop import references_to_documents
from pipeline.output_writer import write_experiments_output
from pipeline.retrieval import ChunkedRetrievalStore, make_embed_fn_litellm, make_embed_fn_local
from formatters import get_create_document_conversation


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG collapse pipeline")

    # -------------------------
    # Model configuration
    # -------------------------
    parser.add_argument(
        "--model-mode",
        choices=["api", "local"],
        required=True,
        help="Run mode for the model",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model identifier",
    )

    # Generation params (applies to both modes)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--top-p", type=float, default=0.9)

    # Local-only knobs (ignored for api mode)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem-util", type=float, default=0.7)
    parser.add_argument(
        "--allow-no-gpu",
        action="store_true",
        help="Allow local mode when CUDA_VISIBLE_DEVICES is empty (e.g. login node). Prefer --model-mode api when no GPU.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (vLLM). Use 2 for 7B on 2 GPUs. vLLM handles GPU distribution internally.",
    )

    # -------------------------
    # Dataset / output
    # -------------------------
    parser.add_argument(
        "--dataset-path",
        default="datasets/umass_data.entity.chatgpt.50.jsonl",
        help="Path to input dataset",
    )
    parser.add_argument(
        "--output-path",
        default="experiment_result_output.json",
        help="Path to write experiment results",
    )

    # -------------------------
    # Experiment parameters
    # -------------------------
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of feedback iterations",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of independent runs per iteration",
    )
    parser.add_argument(
        "--chars-per-doc",
        type=int,
        default=400,
        help="Character limit per document",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions (omit to use all)",
    )
    parser.add_argument(
        "--pipeline-variant",
        choices=list(PIPELINE_VARIANTS),
        default="hybrid",
        help="Variant: hybrid, replace_one (paper: one slot replaced per round), or search",
    )
    parser.add_argument(
        "--search-embedding-mode",
        choices=["local", "api"],
        default="local",
        help="Search variant embeddings: local (SentenceTransformer) or api (LiteLLM). Use local when your API has no embedding models.",
    )
    parser.add_argument(
        "--search-embedding-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model: for local mode = SentenceTransformer name (e.g. all-MiniLM-L6-v2); for api = LiteLLM model name.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Cap on rounds (for testing). If set, uses min(variant default, this value).",
    )
    # Hybrid variant: ratio of synthetic vs original (database) docs
    parser.add_argument(
        "--num-synth-docs",
        type=int,
        default=1,
        help="Number of synthetic docs (from model generations) per round. Use e.g. 10 with --num-db-docs 0 for replace_all-style.",
    )
    parser.add_argument(
        "--num-db-docs",
        type=int,
        default=3,
        help="Number of database docs (from this question's references) per round. Use 0 for all-synthetic (replace_all-style).",
    )
    parser.add_argument(
        "--db-doc-selection",
        choices=["first", "random"],
        default="first",
        help="How to select database docs from references: first or random.",
    )
    parser.add_argument(
        "--synth-doc-selection",
        choices=["first", "random"],
        default="first",
        help="How to select synthetic doc(s) from multiple runs: first or random.",
    )
    parser.add_argument(
        "--stop-if-converged",
        action="store_true",
        help="Stop early when answer signature is identical for four consecutive rounds (no new eval metrics).",
    )

    return parser.parse_args()


@record
def run_pipeline() -> None:
    """
    Generate experiment output JSON.

    Variants:
    - hybrid: context = num_synth_docs (from generations) + num_db_docs (from refs). One pipeline;
      e.g. --num-synth-docs 10 --num-db-docs 0 = replace_all-style; --num-synth-docs 1 --num-db-docs 3 = fixed mix.
    - search: vector retrieval each round (30 rounds).
    """
    args = parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_path
    num_runs = args.num_runs
    chars_per_doc = args.chars_per_doc
    max_questions = args.max_questions
    variant = args.pipeline_variant

    # Round count per variant; optional cap for quick tests
    num_iterations = get_rounds_for_variant(variant)
    if args.max_iterations is not None:
        num_iterations = min(num_iterations, args.max_iterations)

    # Hybrid config (used when variant is hybrid)
    hybrid_config = HybridContextConfig(
        num_synth_docs=args.num_synth_docs,
        num_db_docs=args.num_db_docs,
        db_doc_selection=args.db_doc_selection,
        synth_doc_selection=args.synth_doc_selection,
    )

    # -------------------------
    # Load and prepare dataset (filter min citations)
    # -------------------------
    raw = load_dataset(dataset_path)
    dataset = prepare_dataset(raw, variant)

    # -------------------------
    # Initialize model (CLI-driven)
    # -------------------------
    tp_size = getattr(args, "tensor_parallel_size", 1) or 1
    # vLLM handles tensor parallelism internally (spawns its own processes), so we don't need torch.distributed.run
    # Just pass tensor_parallel_size to vLLM and it will use all GPUs visible via CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print(f"[GPU Config] CUDA_VISIBLE_DEVICES={cuda_devices}, tensor_parallel_size={tp_size}")
    if tp_size > 1:
        import subprocess
        try:
            result = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                gpu_count = len([line for line in result.stdout.strip().split('\n') if 'GPU' in line])
                print(f"[GPU Config] Detected {gpu_count} GPU(s) via nvidia-smi")
        except Exception:
            pass
    llm, resolved_model_name = build_llm(
        model_mode=args.model_mode,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        require_gpu=not args.allow_no_gpu,
        tensor_parallel_size=tp_size,
    )
    print(f"[LLM] Connected. Served model: {getattr(llm, 'served_model_name', resolved_model_name)}", flush=True)

    experiments_metadata: Dict[str, Any] = {
        "model": resolved_model_name,
        "pipeline_variant": variant,
        "num_iterations": num_iterations,
        "num_runs_per_iteration": num_runs,
    }
    if is_hybrid(variant):
        experiments_metadata["num_synth_docs"] = args.num_synth_docs
        experiments_metadata["num_db_docs"] = args.num_db_docs
        experiments_metadata["db_doc_selection"] = args.db_doc_selection
        experiments_metadata["synth_doc_selection"] = args.synth_doc_selection
    experiments: Dict[str, Any] = {
        "experiment_metadata": experiments_metadata,
        "questions": [],
    }

    # Search variant: build embed function (local defaults to HF_HOME cache; no API key needed)
    embed_fn = None
    if is_search(variant):
        if args.search_embedding_mode == "local":
            embed_fn = make_embed_fn_local(model_name=args.search_embedding_model)
        else:
            embed_fn = make_embed_fn_litellm(model=args.search_embedding_model)

    # -------------------------
    # Initialize per-question state
    # -------------------------
    stop_if_converged = getattr(args, "stop_if_converged", False)

    class _QState:
        """Mutable per-question state for the iteration-major loop."""
        __slots__ = (
            "q_idx", "question_text", "references", "current_docs",
            "store", "question_obj", "convergence_signatures", "active",
        )

    states: List[_QState] = []
    for q_idx, row in enumerate(dataset):
        if max_questions is not None and q_idx >= max_questions:
            break

        s = _QState()
        s.q_idx = q_idx
        s.question_text = row["question"]
        s.references = row.get("references", [])
        s.active = True
        s.convergence_signatures = []
        s.question_obj = {
            "question_id": q_idx,
            "question_text": s.question_text,
            "iterations": [],
        }

        if is_search(variant):
            s.store = ChunkedRetrievalStore(embed_fn=embed_fn)
            ref_docs = references_to_documents(s.references, iteration=0)
            s.store.add_documents(ref_docs)
            s.current_docs = s.store.search(s.question_text, k=SEARCH_TOP_K)
        elif is_replace_one(variant):
            s.current_docs = get_initial_documents_replace_one(s.references)
            s.store = None
        else:
            s.current_docs = get_initial_documents_hybrid(s.references, hybrid_config)
            s.store = None

        states.append(s)

    # -------------------------
    # Main experiment loop: iteration-major (batch across all questions)
    # -------------------------
    for it in range(num_iterations):
        active = [s for s in states if s.active]
        if not active:
            break

        # --- Step 1: batch sample_runs across all active questions ---
        # Build one conversation per (question, run) and flatten into a single batch
        batch_conversations: List[List[Dict[str, str]]] = []
        for s in active:
            conversation = build_rag_conversation(
                question=s.question_text,
                docs=s.current_docs,
                chars_per_doc=chars_per_doc,
            )
            batch_conversations.extend([conversation] * num_runs)

        print(f"[Iter {it + 1}/{num_iterations}] Sending {len(batch_conversations)} answer requests ({len(active)} question(s) × {num_runs} runs)...", flush=True)
        all_answers = llm.inference_batch(batch_conversations)
        print(f"[Iter {it + 1}/{num_iterations}] Answer inference complete.", flush=True)

        # Slice answers back to per-question groups
        offset = 0
        per_question_answers: List[List[str]] = []
        for _ in active:
            per_question_answers.append(all_answers[offset : offset + num_runs])
            offset += num_runs

        # --- Step 2: record iteration results and check convergence ---
        still_active_for_docs: List[int] = []  # indices into active[]
        for i, s in enumerate(active):
            answers = per_question_answers[i]
            iteration_obj: Dict[str, Any] = {
                "iteration_number": it,
                "documents": s.current_docs,
                "runs": [
                    {"run_id": f"{it}_{r}", "answer": ans}
                    for r, ans in enumerate(answers)
                ],
            }
            s.question_obj["iterations"].append(iteration_obj)

            # Convergence early stop
            if stop_if_converged:
                sig = tuple(sorted(len(a) for a in answers))
                s.convergence_signatures.append(sig)
                if (
                    len(s.convergence_signatures) >= 4
                    and len(set(s.convergence_signatures[-4:])) == 1
                ):
                    s.active = False
                    continue

            if it < num_iterations - 1:
                still_active_for_docs.append(i)

        # --- Step 3: batch document creation across remaining active questions ---
        if still_active_for_docs:
            doc_batch: List[List[Dict[str, str]]] = []
            runs_per_q: List[int] = []
            for i in still_active_for_docs:
                answers = per_question_answers[i]
                convos = [get_create_document_conversation(content=ans) for ans in answers]
                doc_batch.extend(convos)
                runs_per_q.append(len(convos))

            print(f"[Iter {it + 1}/{num_iterations}] Creating {len(doc_batch)} documents...", flush=True)
            all_doc_texts = llm.inference_batch(doc_batch)
            print(f"[Iter {it + 1}/{num_iterations}] Document creation complete.", flush=True)

            # Slice back and update each question's docs for next iteration
            offset = 0
            for idx, i in enumerate(still_active_for_docs):
                s = active[i]
                n = runs_per_q[idx]
                document_texts = all_doc_texts[offset : offset + n]
                offset += n
                s.current_docs = get_next_documents(
                    variant,
                    s.current_docs,
                    document_texts,
                    iteration=it + 1,
                    references=s.references,
                    question_text=s.question_text,
                    store=s.store,
                    hybrid_config=hybrid_config,
                )

    # Collect results (preserve original question order)
    for s in states:
        experiments["questions"].append(s.question_obj)

    # -------------------------
    # Write output
    # -------------------------
    write_experiments_output(output_path, experiments)

    # Best-effort shutdown
    try:
        llm.shutdown()
    except Exception:
        pass

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    run_pipeline()
