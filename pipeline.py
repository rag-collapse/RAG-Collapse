import argparse
import os
import random
import re
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

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _resolve_citations(
    citation_ids: List[int],
    citation_index: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_id = {entry["citation_id"]: entry for entry in citation_index}
    return [by_id[cid] for cid in citation_ids if cid in by_id]


def _to_token_set(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "") if len(w) >= 4}


def _overlap_score(answer_tokens: set[str], support_tokens: set[str]) -> float:
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & support_tokens) / max(1, len(answer_tokens))


def _answer_change_score(baseline_answer: str, loo_answer: str) -> float:
    """
    Lightweight change score in [0, 1]:
    0 means near-identical lexical content, 1 means no lexical overlap.
    """
    baseline_tokens = _to_token_set(baseline_answer)
    loo_tokens = _to_token_set(loo_answer)
    return 1.0 - _overlap_score(baseline_tokens, loo_tokens)


def _select_top_m_candidate_citation_ids(
    answer: str,
    citation_index: List[Dict[str, Any]],
    docs_by_doc_id: Dict[str, Dict[str, Any]],
    *,
    top_m: int,
) -> List[int]:
    """
    Choose top-M candidate docs by quick lexical overlap with the answer.
    """
    answer_tokens = _to_token_set(answer)
    scored: List[tuple[float, int]] = []
    for entry in citation_index:
        cid = entry.get("citation_id")
        doc = docs_by_doc_id.get(entry.get("doc_id"))
        if cid is None or doc is None:
            continue
        doc_tokens = _to_token_set(doc.get("text", ""))
        score = _overlap_score(answer_tokens, doc_tokens)
        scored.append((score, cid))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [cid for _, cid in scored[: max(1, top_m)]]


def _infer_citation_ids_via_retrieval_loo(
    answer: str,
    citation_index: List[Dict[str, Any]],
    docs_by_doc_id: Dict[str, Dict[str, Any]],
    *,
    rerun_answers_by_citation_id: Dict[int, str],
    top_m: int = 4,
    change_threshold: float = 0.18,
    max_ids: int = 6,
) -> List[int]:
    """
    Wrapper for feasible retrieval-set LOO rerun attribution.
    """
    if not citation_index or not docs_by_doc_id:
        return []
    candidate_ids = _select_top_m_candidate_citation_ids(
        answer=answer,
        citation_index=citation_index,
        docs_by_doc_id=docs_by_doc_id,
        top_m=top_m,
    )
    if not candidate_ids:
        return []

    selected: List[int] = []
    scored_changes: List[tuple[float, int]] = []
    for cid in candidate_ids:
        loo_answer = rerun_answers_by_citation_id.get(cid)
        if not loo_answer:
            continue
        change = _answer_change_score(answer, loo_answer)
        scored_changes.append((change, cid))
        if change >= change_threshold:
            selected.append(cid)
            if len(selected) >= max_ids:
                break

    if selected:
        return selected

    scored_changes.sort(key=lambda x: x[0], reverse=True)
    fallback = [cid for change, cid in scored_changes if change > 0]
    return fallback[: min(2, max_ids)]


def _paraphrase_reference_dataset(
    dataset: List[Dict[str, Any]],
    llm: Any,
) -> List[Dict[str, Any]]:
    """
    Rewrite human-authored reference docs through the same document-creation step used
    for model answers so both document sources share the same surface style.
    """
    conversations: List[List[Dict[str, str]]] = []
    ref_locations: List[tuple[int, int]] = []

    for row_idx, row in enumerate(dataset):
        for ref_idx, ref in enumerate(row.get("references", []) or []):
            text = (ref.get("text") or "").strip()
            if not text:
                continue
            conversations.append(get_create_document_conversation(content=text))
            ref_locations.append((row_idx, ref_idx))

    if not conversations:
        return dataset

    paraphrased_texts = llm.inference_batch(conversations)
    rewritten_dataset = []
    for row in dataset:
        rewritten_dataset.append({**row, "references": [dict(ref) for ref in row.get("references", []) or []]})

    for (row_idx, ref_idx), paraphrased_text in zip(ref_locations, paraphrased_texts):
        rewritten_dataset[row_idx]["references"][ref_idx]["text"] = paraphrased_text

    return rewritten_dataset


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
    parser.add_argument(
        "--doc-model-mode",
        choices=["api", "local"],
        default=None,
        help="Optional separate model mode for document rewriting/paraphrasing. Defaults to --model-mode.",
    )
    parser.add_argument(
        "--doc-model-name",
        default=None,
        help="Optional separate model identifier for document rewriting/paraphrasing. Defaults to --model-name.",
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
        "--paraphrase-reference-docs",
        action="store_true",
        help="Rewrite human-written reference docs through the create-document prompt before iteration 0.",
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
    parser.add_argument(
        "--enable-citations",
        action="store_true",
        help="Enable citation generation using retrieval_loo; otherwise citations are disabled.",
    )
    parser.add_argument(
        "--citation-max-docs",
        type=int,
        default=6,
        help="Maximum number of citations to keep per run.",
    )
    parser.add_argument(
        "--citation-top-m",
        type=int,
        default=4,
        help="Top-M retrieval docs (by quick lexical overlap) to test with leave-one-out reruns.",
    )
    parser.add_argument(
        "--citation-change-threshold",
        type=float,
        default=0.18,
        help="Minimum answer-change score to keep a doc as a citation after leave-one-out rerun.",
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
    paraphrase_reference_docs = bool(args.paraphrase_reference_docs)
    max_questions = args.max_questions
    variant = args.pipeline_variant
    citations_enabled = bool(args.enable_citations)
    citation_max_docs = max(1, int(getattr(args, "citation_max_docs", 6)))
    citation_top_m = max(1, int(getattr(args, "citation_top_m", 4)))
    citation_change_threshold = float(getattr(args, "citation_change_threshold", 0.18))

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
    doc_model_mode = args.doc_model_mode or args.model_mode
    doc_model_name = args.doc_model_name or args.model_name
    if doc_model_mode == args.model_mode and doc_model_name == args.model_name:
        doc_llm = llm
        resolved_doc_model_name = resolved_model_name
    else:
        doc_llm, resolved_doc_model_name = build_llm(
            model_mode=doc_model_mode,
            model_name=doc_model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            require_gpu=not args.allow_no_gpu,
            tensor_parallel_size=tp_size,
        )

    experiments_metadata: Dict[str, Any] = {
        "model": resolved_model_name,
        "doc_model": resolved_doc_model_name,
        "pipeline_variant": variant,
        "num_iterations": num_iterations,
        "num_runs_per_iteration": num_runs,
        "citations_enabled": citations_enabled,
        "reference_docs_paraphrased": paraphrase_reference_docs,
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

    if paraphrase_reference_docs:
        # Normalize the surface form of human-written references using the same
        # document-creation step applied to model-generated answers.
        dataset = _paraphrase_reference_dataset(
            dataset,
            doc_llm,
        )

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

    # Cache LOO rerun answers keyed by (question_id, iteration_number, removed_citation_id).
    loo_rerun_cache: Dict[tuple[int, int, int], str] = {}

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
        prompt_docs_by_question: List[List[Dict[str, Any]]] = []
        for s in active:
            prompt_docs = list(s.current_docs)
            if citations_enabled and len(prompt_docs) > 1:
                random.shuffle(prompt_docs)
            conversation = build_rag_conversation(
                question=s.question_text,
                docs=prompt_docs,
                chars_per_doc=chars_per_doc,
                shuffle_docs=not citations_enabled,
            )
            prompt_docs_by_question.append(prompt_docs)
            batch_conversations.extend([conversation] * num_runs)

        all_answers = llm.inference_batch(batch_conversations)

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
            prompt_docs = prompt_docs_by_question[i]
            citation_index: List[Dict[str, Any]] = []
            docs_by_doc_id: Dict[str, Dict[str, Any]] = {}
            citation_entry_by_id: Dict[int, Dict[str, Any]] = {}
            if citations_enabled:
                citation_index = [
                    {
                        "citation_id": j + 1,
                        "doc_id": doc.get("doc_id"),
                        "url": doc.get("url", ""),
                        "iteration": doc.get("iteration"),
                    }
                    for j, doc in enumerate(prompt_docs)
                ]
                docs_by_doc_id = {
                    d.get("doc_id"): d for d in prompt_docs if d.get("doc_id")
                }
                citation_entry_by_id = {
                    e["citation_id"]: e for e in citation_index if "citation_id" in e
                }

            # Build/lookup cached leave-one-out rerun answers for top-M candidate docs.
            rerun_answers_by_citation_id: Dict[int, str] = {}
            if citations_enabled:
                needed_cids: set[int] = set()
                for ans in answers:
                    needed_cids.update(
                        _select_top_m_candidate_citation_ids(
                            answer=ans,
                            citation_index=citation_index,
                            docs_by_doc_id=docs_by_doc_id,
                            top_m=citation_top_m,
                        )
                    )

                missing_keys: List[tuple[int, int, int]] = []
                missing_conversations: List[List[Dict[str, str]]] = []
                for cid in sorted(needed_cids):
                    cache_key = (s.q_idx, it, cid)
                    if cache_key in loo_rerun_cache:
                        continue
                    entry = citation_entry_by_id.get(cid)
                    if not entry:
                        continue
                    removed_doc_id = entry.get("doc_id")
                    docs_without = [
                        d for d in prompt_docs if d.get("doc_id") != removed_doc_id
                    ]
                    if not docs_without:
                        continue
                    loo_conversation = build_rag_conversation(
                        question=s.question_text,
                        docs=docs_without,
                        chars_per_doc=chars_per_doc,
                        shuffle_docs=False,
                    )
                    missing_keys.append(cache_key)
                    missing_conversations.append(loo_conversation)

                if missing_conversations:
                    missing_outputs = llm.inference_batch(missing_conversations)
                    for key, out in zip(missing_keys, missing_outputs):
                        loo_rerun_cache[key] = out

                for cid in needed_cids:
                    cache_key = (s.q_idx, it, cid)
                    if cache_key in loo_rerun_cache:
                        rerun_answers_by_citation_id[cid] = loo_rerun_cache[cache_key]

            runs = []
            for r, ans in enumerate(answers):
                if not citations_enabled:
                    citation_ids = []
                else:
                    citation_ids = _infer_citation_ids_via_retrieval_loo(
                        ans,
                        citation_index,
                        docs_by_doc_id,
                        rerun_answers_by_citation_id=rerun_answers_by_citation_id,
                        top_m=citation_top_m,
                        change_threshold=citation_change_threshold,
                        max_ids=citation_max_docs,
                    )
                runs.append(
                    {
                        "run_id": f"{it}_{r}",
                        "answer": ans,
                        "citation_ids": citation_ids,
                        "citations": _resolve_citations(citation_ids, citation_index)
                        if citations_enabled
                        else [],
                    }
                )

            iteration_obj: Dict[str, Any] = {
                "iteration_number": it,
                "documents": s.current_docs,
                "runs": runs,
            }
            if citations_enabled:
                iteration_obj["citation_index"] = citation_index
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
                if is_replace_one(variant) or is_search(variant):
                    answers = [random.choice(answers)]
                convos = [get_create_document_conversation(content=ans) for ans in answers]
                doc_batch.extend(convos)
                runs_per_q.append(len(convos))

            all_doc_texts = doc_llm.inference_batch(doc_batch)

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
    if doc_llm is not llm:
        try:
            doc_llm.shutdown()
        except Exception:
            pass

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    run_pipeline()
