import argparse
import os
from typing import Any, Dict, List

from pipeline.config import (
    PIPELINE_VARIANTS,
    get_rounds_for_variant,
    is_hybrid,
    is_search,
    SEARCH_TOP_K,
)
from pipeline.context_builder import (
    HybridContextConfig,
    get_initial_documents_hybrid,
    get_next_documents,
)
from pipeline.data_loader import load_dataset, prepare_dataset
from pipeline.prompt_builder import build_rag_conversation
from pipeline.model_runner import build_llm, sample_runs
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
        help="Variant: hybrid (configurable synth/db ratio) or search (retrieval)",
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

    return parser.parse_args()


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
    )

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
    # Main experiment loop (single path: context_builder for next docs)
    # -------------------------
    for q_idx, row in enumerate(dataset):
        if max_questions is not None and q_idx >= max_questions:
            break

        question_text: str = row["question"]
        references: List[Dict[str, Any]] = row.get("references", [])

        if is_search(variant):
            store = ChunkedRetrievalStore(embed_fn=embed_fn)
            ref_docs = references_to_documents(references, iteration=0)
            store.add_documents(ref_docs)
            current_docs = store.search(question_text, k=SEARCH_TOP_K)
        else:
            current_docs = get_initial_documents_hybrid(references, hybrid_config)
            store = None

        question_obj: Dict[str, Any] = {
            "question_id": q_idx,
            "question_text": question_text,
            "iterations": [],
        }

        for it in range(num_iterations):
            conversation = build_rag_conversation(
                question=question_text,
                docs=current_docs,
                chars_per_doc=chars_per_doc,
            )

            answers = sample_runs(
                llm=llm,
                conversation=conversation,
                num_runs=num_runs,
            )

            iteration_obj: Dict[str, Any] = {
                "iteration_number": it,
                "documents": current_docs,
                "runs": [
                    {"run_id": f"{it}_{i}", "answer": ans}
                    for i, ans in enumerate(answers)
                ],
            }

            question_obj["iterations"].append(iteration_obj)

            # Next iteration docs: single call to context_builder
            if it < num_iterations - 1:
                doc_conversations = [get_create_document_conversation(content=ans) for ans in answers]
                document_texts = llm.inference_batch(doc_conversations)
                current_docs = get_next_documents(
                    variant,
                    current_docs,
                    document_texts,
                    iteration=it + 1,
                    references=references,
                    question_text=question_text,
                    store=store,
                    hybrid_config=hybrid_config,
                )

        experiments["questions"].append(question_obj)

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
