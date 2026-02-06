import os
from typing import Any, Dict, List

from pipeline.data_loader import load_dataset
from pipeline.prompt_builder import build_rag_conversation
from pipeline.model_runner import build_llm_from_env, sample_runs
from pipeline.feedback_loop import references_to_documents, answers_to_documents
from pipeline.output_writer import write_experiments_output


def run_pipeline() -> None:
    """
    Generate example_experiments_output.json.

    This pipeline simulates RAG collapse by repeatedly feeding
    model-generated answers back as documents.

    Environment variables:
      DATASET_PATH: input JSONL or JSON dataset
      OUTPUT_PATH: output JSON file
      MODEL_MODE: api | local
      MODEL_NAME: model identifier
      NUM_ITERATIONS: number of collapse iterations
      NUM_RUNS: number of samples per iteration
      CHARS_PER_DOC: document truncation length
      MAX_QUESTIONS: optional limit for debugging
    """

    # -------------------------
    # Environment configuration
    # -------------------------
    dataset_path = os.getenv(
        "DATASET_PATH",
        "datasets/umass_data.entity.chatgpt.50.jsonl",
    )
    output_path = os.getenv(
        "OUTPUT_PATH",
        "example_experiments_output.json",
    )

    num_iterations = int(os.getenv("NUM_ITERATIONS", "5"))
    num_runs = int(os.getenv("NUM_RUNS", "10"))
    chars_per_doc = int(os.getenv("CHARS_PER_DOC", "400"))

    max_questions_env = os.getenv("MAX_QUESTIONS", "").strip()
    max_questions = int(max_questions_env) if max_questions_env else None

    # -------------------------
    # Load dataset
    # -------------------------
    dataset = load_dataset(dataset_path)

    # -------------------------
    # Initialize model
    # -------------------------
    llm, resolved_model_name = build_llm_from_env()

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
            # Build RAG-style conversation using existing formatter logic
            conversation = build_rag_conversation(
                question=question_text,
                docs=current_docs,
                chars_per_doc=chars_per_doc,
            )

            # Sample multiple independent runs
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
                current_docs = answers_to_documents(
                    answers,
                    iteration=it + 1,
                )

        experiments["questions"].append(question_obj)

    # -------------------------
    # Write output
    # -------------------------
    write_experiments_output(output_path, experiments)

    # Best-effort shutdown (important for GPU / vLLM)
    try:
        llm.shutdown()
    except Exception:
        pass

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    run_pipeline()
