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
    is_agentic_rag,
    SEARCH_TOP_K,
    SEARCH_CHUNK_SIZE,
    SEARCH_CHUNK_OVERLAP,
    AGENTIC_MAX_TOOL_CALLS,
)
from pipeline.context_builder import (
    HybridContextConfig,
    get_initial_documents_hybrid,
    get_initial_documents_replace_one,
    get_next_documents,
)
from pipeline.data_loader import load_dataset, prepare_dataset
from pipeline.prompt_builder import build_rag_conversation, build_agentic_rag_conversation
from pipeline.model_runner import build_llm
from pipeline.feedback_loop import references_to_documents
from pipeline.output_writer import write_experiments_output
from pipeline.retrieval import ChunkedRetrievalStore, make_embed_fn_litellm, make_embed_fn_local
from formatters import get_create_document_conversation, get_context_str_from_docs, RETRIEVE_TOOL_SPEC

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
            conversations.append(get_create_document_conversation(question=row.get("question", ""), answer=text))
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


def _make_retrieve_executor(store, k: int, chars_per_doc: int, retrieved_chunks: list = None):
    """
    Return a tool_executor callable for the agentic_rag retrieve tool.
    Searches the given ChunkedRetrievalStore and formats results as 'Context N:' strings,
    matching the format used by build_rag_conversation so the model sees consistent context.

    If retrieved_chunks is provided (a list passed by reference), every chunk returned
    by a retrieve call is appended to it for post-hoc citation tracking.
    """
    def execute(name: str, args: dict) -> str:
        if name == "retrieve":
            query = args.get("query", "")
            chunks = store.search(query, k=k)
            if retrieved_chunks is not None:
                retrieved_chunks.extend(chunks)
            return get_context_str_from_docs(chunks, chars_per_doc=chars_per_doc, shuffle=False)
        return f"Unknown tool: {name}"
    return execute


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG collapse pipeline")

    # -------------------------
    # Model configuration
    # -------------------------
    parser.add_argument(
        "--model-mode",
        choices=["api", "local", "server"],
        required=True,
        help="Run mode: 'api' (LiteLLM/proprietary), 'local' (in-process vLLM), 'server' (HTTP client to vLLM server)",
    )
    parser.add_argument(
        "--vllm-api-base",
        default=None,
        help="vLLM server base URL (required for --model-mode server), e.g. http://host:5150/v1",
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

    # -------------------------
    # Document generation model (optional; falls back to main LLM if not set)
    # -------------------------
    parser.add_argument(
        "--doc-model-mode",
        choices=["api", "local", "server"],
        default=None,
        help="Model mode for document generation. Omit to reuse the main model.",
    )
    parser.add_argument(
        "--doc-vllm-api-base",
        default=None,
        help="vLLM server base URL for doc generation (required when --doc-model-mode server).",
    )
    parser.add_argument(
        "--doc-model-name",
        default=None,
        help="Model identifier for doc generation. Defaults to --model-name.",
    )
    parser.add_argument("--doc-temperature", type=float, default=None,
        help="Temperature for doc generation. Defaults to --temperature.")
    parser.add_argument("--doc-max-tokens", type=int, default=None,
        help="Max tokens for doc generation. Defaults to --max-tokens.")
    parser.add_argument("--doc-top-p", type=float, default=None,
        help="Top-p for doc generation. Defaults to --top-p.")

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
        "--search-top-k",
        type=int,
        default=SEARCH_TOP_K,
        help="Number of top chunks to retrieve per round in the search variant (default: %(default)s).",
    )
    parser.add_argument(
        "--search-chunk-size",
        type=int,
        default=SEARCH_CHUNK_SIZE,
        help="Character size of each text chunk when indexing documents in the search variant (default: %(default)s).",
    )
    parser.add_argument(
        "--search-chunk-overlap",
        type=int,
        default=SEARCH_CHUNK_OVERLAP,
        help="Character overlap between consecutive chunks in the search variant (default: %(default)s).",
    )
    parser.add_argument(
        "--agentic-max-tool-calls",
        type=int,
        default=AGENTIC_MAX_TOOL_CALLS,
        help="Maximum retrieve tool calls per answer in the agentic_rag variant (default: %(default)s).",
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

    if args.model_mode == "server" and not args.vllm_api_base:
        raise SystemExit("ERROR: --vllm-api-base is required when --model-mode=server")
    if args.doc_model_mode == "server" and not args.doc_vllm_api_base:
        raise SystemExit("ERROR: --doc-vllm-api-base is required when --doc-model-mode=server")

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

    # Search retrieval settings (used only for search variant)
    search_top_k = args.search_top_k
    search_chunk_size = args.search_chunk_size
    search_chunk_overlap = args.search_chunk_overlap
    agentic_max_tool_calls = args.agentic_max_tool_calls

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
    if args.model_mode != "server":
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
        api_base=args.vllm_api_base,
    )
    print(f"[LLM] Connected. Served model: {getattr(llm, 'served_model_name', resolved_model_name)}", flush=True)

    if args.doc_model_mode is not None:
        doc_llm, doc_model_name = build_llm(
            model_mode=args.doc_model_mode,
            model_name=args.doc_model_name or args.model_name,
            temperature=args.doc_temperature if args.doc_temperature is not None else args.temperature,
            max_tokens=args.doc_max_tokens if args.doc_max_tokens is not None else args.max_tokens,
            top_p=args.doc_top_p if args.doc_top_p is not None else args.top_p,
            api_base=args.doc_vllm_api_base,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            require_gpu=not args.allow_no_gpu,
            tensor_parallel_size=tp_size,
        )
        print(f"[Doc LLM] Connected. Served model: {getattr(doc_llm, 'served_model_name', doc_model_name)}", flush=True)
    else:
        doc_llm = llm
        doc_model_name = resolved_model_name

    experiments_metadata: Dict[str, Any] = {
        "model": resolved_model_name,
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
    if args.doc_model_mode is not None:
        experiments_metadata["doc_model"] = doc_model_name
        experiments_metadata["doc_model_mode"] = args.doc_model_mode
    if is_search(variant) or is_agentic_rag(variant):
        experiments_metadata["search_top_k"] = search_top_k
        experiments_metadata["search_chunk_size"] = search_chunk_size
        experiments_metadata["search_chunk_overlap"] = search_chunk_overlap
    if is_agentic_rag(variant):
        experiments_metadata["agentic_max_tool_calls"] = agentic_max_tool_calls
    experiments: Dict[str, Any] = {
        "experiment_metadata": experiments_metadata,
        "questions": [],
    }

    # Search variant: build embed function (local defaults to HF_HOME cache; no API key needed)
    embed_fn = None
    if is_search(variant) or is_agentic_rag(variant):
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

    total_questions = min(max_questions, len(dataset)) if max_questions is not None else len(dataset)
    if is_search(variant) or is_agentic_rag(variant):
        print(f"[Init] Building retrieval stores for {total_questions} question(s) (embedding on CPU — this may take a few minutes)...", flush=True)

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

        if is_search(variant) or is_agentic_rag(variant):
            s.store = ChunkedRetrievalStore(
                embed_fn=embed_fn,
                chunk_size=search_chunk_size,
                chunk_overlap=search_chunk_overlap,
            )
            ref_docs = references_to_documents(s.references, iteration=0)
            s.store.add_documents(ref_docs)
            s.current_docs = s.store.search(s.question_text, k=search_top_k)
            if (q_idx + 1) % 50 == 0 or (q_idx + 1) == total_questions:
                print(f"[Init] Embedded {q_idx + 1}/{total_questions} question(s).", flush=True)
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
        prompt_docs_by_question: List[List[Dict[str, Any]]] = []

        if is_agentic_rag(variant):
            # Agentic flow: model retrieves its own context via the retrieve tool.
            # No documents are pre-injected into the prompt.
            agentic_conversations: List[List[Dict[str, str]]] = []
            agentic_executors = []
            # per_question_run_chunks[q][r] = list of chunks retrieved during run r of question q
            per_question_run_chunks: List[List[List[Dict[str, Any]]]] = []
            for s in active:
                conv = build_agentic_rag_conversation(s.question_text)
                run_chunks_for_q: List[List[Dict[str, Any]]] = []
                for _ in range(num_runs):
                    run_chunks: List[Dict[str, Any]] = []
                    executor_fn = _make_retrieve_executor(s.store, search_top_k, chars_per_doc, run_chunks)
                    agentic_conversations.append(conv)
                    agentic_executors.append(executor_fn)
                    run_chunks_for_q.append(run_chunks)
                per_question_run_chunks.append(run_chunks_for_q)
                prompt_docs_by_question.append([])  # no pre-fetched docs

            print(
                f"[Iter {it + 1}/{num_iterations}] Sending {len(agentic_conversations)} agentic answer requests "
                f"({len(active)} question(s) × {num_runs} runs, max {agentic_max_tool_calls} tool calls each)...",
                flush=True,
            )
            agentic_results = llm.inference_agentic_batch(
                agentic_conversations,
                tools=[RETRIEVE_TOOL_SPEC],
                tool_executors=agentic_executors,
                max_tool_calls=agentic_max_tool_calls,
            )
            all_answers = [ans for ans, _ in agentic_results]
            all_tool_calls_used = [n for _, n in agentic_results]
            print(f"[Iter {it + 1}/{num_iterations}] Agentic inference complete.", flush=True)

        else:
            # Standard flow: pre-fetch docs and inject into prompt.
            batch_conversations: List[List[Dict[str, str]]] = []
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

            print(
                f"[Iter {it + 1}/{num_iterations}] Sending {len(batch_conversations)} answer requests "
                f"({len(active)} question(s) × {num_runs} runs)...",
                flush=True,
            )
            all_answers = llm.inference_batch(batch_conversations)
            print(f"[Iter {it + 1}/{num_iterations}] Answer inference complete.", flush=True)

        # Slice answers (and tool call counts for agentic) back to per-question groups
        offset = 0
        per_question_answers: List[List[str]] = []
        per_question_tool_calls: List[List[int]] = []
        _tool_calls_src = all_tool_calls_used if is_agentic_rag(variant) else None
        for _ in active:
            per_question_answers.append(all_answers[offset : offset + num_runs])
            per_question_tool_calls.append(
                _tool_calls_src[offset : offset + num_runs]
                if _tool_calls_src is not None
                else [0] * num_runs
            )
            offset += num_runs

        # --- Step 2: record iteration results and check convergence ---
        if not is_agentic_rag(variant):
            per_question_run_chunks = [[] for _ in active]
        still_active_for_docs: List[int] = []  # indices into active[]
        for i, s in enumerate(active):
            answers = per_question_answers[i]
            run_tool_calls = per_question_tool_calls[i]
            prompt_docs = prompt_docs_by_question[i]
            citation_index: List[Dict[str, Any]] = []
            docs_by_doc_id: Dict[str, Dict[str, Any]] = {}
            citation_entry_by_id: Dict[int, Dict[str, Any]] = {}
            # For agentic_rag, use docs retrieved via tool calls instead of prompt docs.
            # Aggregate unique doc_ids across all runs, look up full docs from s.current_docs.
            if citations_enabled and is_agentic_rag(variant):
                run_chunks_for_q = per_question_run_chunks[i]
                seen_doc_ids: set = set()
                for run_chunks in run_chunks_for_q:
                    for chunk in run_chunks:
                        seen_doc_ids.add(chunk.get("doc_id", ""))
                seen_doc_ids.discard("")
                current_docs_by_id = {d.get("doc_id"): d for d in s.current_docs if d.get("doc_id")}
                prompt_docs = [current_docs_by_id[did] for did in seen_doc_ids if did in current_docs_by_id]
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
            for r, (ans, n_tool_calls) in enumerate(zip(answers, run_tool_calls)):
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
                run_obj: Dict[str, Any] = {
                    "run_id": f"{it}_{r}",
                    "answer": ans,
                    "citation_ids": citation_ids,
                    "citations": _resolve_citations(citation_ids, citation_index)
                    if citations_enabled
                    else [],
                }
                if is_agentic_rag(variant):
                    run_obj["tool_calls_used"] = n_tool_calls
                runs.append(run_obj)

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
                if is_replace_one(variant) or is_search(variant) or is_agentic_rag(variant):
                    answers = [random.choice(answers)]
                convos = [get_create_document_conversation(question=active[i].question_text, answer=ans) for ans in answers]
                doc_batch.extend(convos)
                runs_per_q.append(len(convos))

            print(f"[Iter {it + 1}/{num_iterations}] Creating {len(doc_batch)} documents...", flush=True)
            all_doc_texts = doc_llm.inference_batch(doc_batch)
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
                    search_top_k=search_top_k,
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
