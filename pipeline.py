import argparse
import os
from typing import Any, Dict, List

from pipeline.data_loader import load_dataset
from pipeline.prompt_builder import build_rag_conversation
from pipeline.model_runner import build_llm, sample_runs
from pipeline.feedback_loop import references_to_documents, answers_to_documents
from pipeline.output_writer import write_experiments_output
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
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--top-p", type=float, default=0.9)

    # Local-only knobs (ignored for api mode)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--cache-dir", type=str, default="model_cache")

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

    return parser.parse_args()


def run_pipeline() -> None:
    """
    Generate experiment output JSON.

    This pipeline simulates RAG collapse by repeatedly feeding
    model-generated answers back as documents.
    """
    args = parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_path
    num_iterations = args.num_iterations
    num_runs = args.num_runs
    chars_per_doc = args.chars_per_doc
    max_questions = args.max_questions
    cache_dir = args.cache_dir

    # -------------------------
    # Load dataset
    # -------------------------
    dataset = load_dataset(dataset_path)

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
        cache_dir=cache_dir,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )

    experiments: Dict[str, Any] = {
        "experiment_metadata": {
            "model": resolved_model_name,
            "num_iterations": num_iterations,
            "num_runs_per_iteration": num_runs,
        },
        "questions": [],
    }

    # -------------------------
    # Main experiment loop
    # -------------------------
    for q_idx, row in enumerate(dataset):
        if max_questions is not None and q_idx >= max_questions:
            break

        question_text: str = row["question"]
        references: List[Dict[str, Any]] = row.get("references", [])

        # Iteration 0 documents come from reference texts
        current_docs = references_to_documents(references, iteration=0)

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

            # Prepare documents for next iteration (collapse mechanism)
            if it < num_iterations - 1:
                # convert
                doc_conversations = [get_create_document_conversation(content=ans) for ans in answers]

                document_texts = llm.inference_batch(doc_conversations)

                current_docs = answers_to_documents(
                    document_texts,
                    iteration=it + 1,
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
