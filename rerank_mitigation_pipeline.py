import argparse
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch

from ai_detection_test import load_detector
from formatters import get_create_document_conversation
from pipeline.config import SEARCH_CHUNK_OVERLAP, SEARCH_CHUNK_SIZE, SEARCH_TOP_K, ROUNDS_SEARCH
from pipeline.data_loader import load_dataset, prepare_dataset
from pipeline.feedback_loop import answers_to_documents, references_to_documents
from pipeline.model_runner import build_llm
from pipeline.output_writer import write_experiments_output
from pipeline.prompt_builder import build_rag_conversation
from pipeline.retrieval import ChunkedRetrievalStore, make_embed_fn_litellm, make_embed_fn_local

def rerank_with_ai_penalty(
    chunks: List[Dict[str, Any]],
    scores: np.ndarray,
    ai_likelihoods: np.ndarray,
    lam: float,
    k: int,
) -> List[Dict[str, Any]]:
    """
    Re-rank retrieved chunks by penalising AI-generated-looking text.
    """
    if not chunks:
        return []

    scores = np.asarray(scores, dtype=np.float32)
    ai_likelihoods = np.asarray(ai_likelihoods, dtype=np.float32)

    # Min-max normalise retrieval scores to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 1e-9:
        norm = (scores - s_min) / (s_max - s_min)
    else:
        norm = np.ones_like(scores)

    penalised = norm * (1.0 - lam * ai_likelihoods)

    order = np.argsort(-penalised)
    seen: set = set()
    results: List[Dict[str, Any]] = []
    for idx in order:
        did = chunks[idx].get("doc_id")
        if did in seen:
            continue
        seen.add(did)
        results.append(chunks[idx])
        if len(results) >= k:
            break
    return results

class ScoredChunkedRetrievalStore(ChunkedRetrievalStore):
    def search_with_scores(self, query: str, k: int = SEARCH_TOP_K):
        """Return (chunks, scores) — top-k by cosine similarity."""
        if not self._chunks:
            return [], np.array([], dtype=np.float32)
        qvec = self._to_qvec(query)
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        v_norm = self._vectors / norms
        q_norm = qvec / (np.linalg.norm(qvec, axis=1, keepdims=True) or 1)
        scores = np.dot(v_norm, q_norm.T).flatten()
        top_indices = np.argsort(scores)[::-1][:k]
        return [self._chunks[i] for i in top_indices], scores[top_indices]

    def _to_qvec(self, query: str) -> np.ndarray:
        from pipeline.retrieval import _to_float32
        return _to_float32(self.embed_fn([query])).reshape(1, -1)

def parse_args():
    p = argparse.ArgumentParser(
        description="RAG-collapse pipeline with AI-likelihood re-ranking (search variant)")

    p.add_argument("--model-mode", choices=["api", "local", "server"], required=True)
    p.add_argument("--vllm-api-base", default=None,
        help="vLLM server base URL (required for --model-mode server)")
    p.add_argument("--model-name", required=True, help="Model identifier")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--top-p", type=float, default=0.9)

    p.add_argument("--doc-model-mode", choices=["api", "local", "server"], default=None)
    p.add_argument("--doc-vllm-api-base", default=None)
    p.add_argument("--doc-model-name", default=None)
    p.add_argument("--doc-temperature", type=float, default=None)
    p.add_argument("--doc-max-tokens", type=int, default=None)
    p.add_argument("--doc-top-p", type=float, default=None)

    # Local-only knobs
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.7)
    p.add_argument("--allow-no-gpu", action="store_true")
    p.add_argument("--tensor-parallel-size", type=int, default=1)

    p.add_argument("--dataset-path", default="datasets/umass_data.entity.chatgpt.50.jsonl")
    p.add_argument("--output-path", default="rerank_experiment_output.json")
    p.add_argument("--max-questions", type=int, default=None)

    p.add_argument("--num-iterations", type=int, default=None,
        help=f"Override default round count (default: {ROUNDS_SEARCH}).")
    p.add_argument("--num-runs", type=int, default=10)
    p.add_argument("--chars-per-doc", type=int, default=400)

    p.add_argument("--search-embedding-mode", choices=["local", "api"], default="local")
    p.add_argument("--search-embedding-model", default="all-MiniLM-L6-v2")
    p.add_argument("--search-top-k", type=int, default=SEARCH_TOP_K)
    p.add_argument("--search-chunk-size", type=int, default=SEARCH_CHUNK_SIZE)
    p.add_argument("--search-chunk-overlap", type=int, default=SEARCH_CHUNK_OVERLAP)

    p.add_argument("--lambda-penalty", type=float, default=0.5,
        help="Penalty weight for AI-likelihood. S2 = S_norm * (1 - lambda * AI_likelihood). "
             "0 = no penalty, 1 = full penalty.")
    p.add_argument("--detector-model", type=str, default="desklib-finetuned",
        help="AI detector model name from MODEL_REGISTRY in ai_detection_test.py.")
    p.add_argument("--detector-batch-size", type=int, default=16)

    return p.parse_args()

def run_pipeline() -> None:
    args = parse_args()

    if args.model_mode == "server" and not args.vllm_api_base:
        raise SystemExit("ERROR: --vllm-api-base is required when --model-mode=server")
    if args.doc_model_mode == "server" and not args.doc_vllm_api_base:
        raise SystemExit("ERROR: --doc-vllm-api-base is required when --doc-model-mode=server")

    num_iterations = args.num_iterations or ROUNDS_SEARCH
    num_runs = args.num_runs
    chars_per_doc = args.chars_per_doc
    lam = args.lambda_penalty
    search_top_k = args.search_top_k

    det_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading AI detector: {args.detector_model} ...")
    detector = load_detector(args.detector_model)
    detector.to(det_device)
    detector.eval()

    def score_ai_likelihood(texts: List[str]) -> List[float]:
        """Return AI-likelihood scores in [0, 1] for a list of texts."""
        all_scores: List[float] = []
        for i in range(0, len(texts), args.detector_batch_size):
            batch = texts[i:i + args.detector_batch_size]
            all_scores.extend(detector.predict_batch(batch, det_device))
        return all_scores

    raw = load_dataset(args.dataset_path)
    dataset = prepare_dataset(raw, "search")

    if args.search_embedding_mode == "local":
        embed_fn = make_embed_fn_local(model_name=args.search_embedding_model)
    else:
        embed_fn = make_embed_fn_litellm(model=args.search_embedding_model)

    tp_size = args.tensor_parallel_size or 1
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

    meta: Dict[str, Any] = {
        "model": resolved_model_name,
        "doc_model": doc_model_name,
        "pipeline_variant": "search_rerank",
        "num_iterations": num_iterations,
        "num_runs_per_iteration": num_runs,
        "search_top_k": search_top_k,
        "search_chunk_size": args.search_chunk_size,
        "search_chunk_overlap": args.search_chunk_overlap,
        "lambda_penalty": lam,
        "detector_model": args.detector_model,
    }
    experiments: Dict[str, Any] = {"experiment_metadata": meta, "questions": []}

    max_questions = args.max_questions
    total_questions = min(max_questions, len(dataset)) if max_questions is not None else len(dataset)

    class _QState:
        __slots__ = (
            "q_idx", "question_text", "references",
            "current_docs", "store", "question_obj",
        )

    # We fetch more than top_k so the re-ranker has candidates to work with
    fetch_k = max(search_top_k * 2, 20)

    print(f"[Init] Building retrieval stores for {total_questions} question(s)...", flush=True)
    states: List[_QState] = []
    for q_idx, row in enumerate(dataset):
        if max_questions is not None and q_idx >= max_questions:
            break

        s = _QState()
        s.q_idx = q_idx
        s.question_text = row["question"]
        s.references = row.get("references", [])

        s.store = ScoredChunkedRetrievalStore(
            embed_fn=embed_fn,
            chunk_size=args.search_chunk_size,
            chunk_overlap=args.search_chunk_overlap,
        )
        ref_docs = references_to_documents(s.references, iteration=0)
        s.store.add_documents(ref_docs)

        # Initial retrieval with re-ranking
        chunks, scores = s.store.search_with_scores(s.question_text, k=fetch_k)
        if chunks:
            ai_scores = score_ai_likelihood([c["text"] for c in chunks])
            s.current_docs = rerank_with_ai_penalty(
                chunks, scores, np.array(ai_scores), lam, search_top_k,
            )
        else:
            s.current_docs = []

        s.question_obj = {
            "question_id": q_idx,
            "question_text": s.question_text,
            "iterations": [],
        }
        states.append(s)
        if (q_idx + 1) % 50 == 0 or (q_idx + 1) == total_questions:
            print(f"[Init] Embedded {q_idx + 1}/{total_questions} question(s).", flush=True)

    for it in range(num_iterations):
        print(f"\n=== Iteration {it}/{num_iterations} (search_rerank, λ={lam}) ===")

        # --- Step 1: batch answer generation ---
        batch_conversations: List[List[Dict[str, str]]] = []
        for s in states:
            conv = build_rag_conversation(
                question=s.question_text,
                docs=s.current_docs,
                chars_per_doc=chars_per_doc,
                shuffle_docs=True,
            )
            batch_conversations.extend([conv] * num_runs)

        print(f"[Iter {it}/{num_iterations}] Sending {len(batch_conversations)} answer requests "
              f"({len(states)} question(s) × {num_runs} runs)...", flush=True)
        all_answers = llm.inference_batch(batch_conversations)
        print(f"[Iter {it}/{num_iterations}] Answer inference complete.", flush=True)

        offset = 0
        per_q_answers: List[List[str]] = []
        for _ in states:
            per_q_answers.append(all_answers[offset:offset + num_runs])
            offset += num_runs

        # --- Step 2: record iteration ---
        need_doc_gen = it < num_iterations - 1
        for i, s in enumerate(states):
            answers = per_q_answers[i]
            runs = [{"run_id": f"{it}_{r}", "answer": ans} for r, ans in enumerate(answers)]
            s.question_obj["iterations"].append({
                "iteration_number": it,
                "documents": s.current_docs,
                "runs": runs,
            })

        if not need_doc_gen:
            continue

        # --- Step 3: batch document creation ---
        doc_batch: List[List[Dict[str, str]]] = []
        for i, s in enumerate(states):
            answers = per_q_answers[i]
            ans_for_doc = random.choice(answers)
            doc_batch.append(get_create_document_conversation(
                question=s.question_text, answer=ans_for_doc,
            ))

        print(f"[Iter {it}/{num_iterations}] Creating {len(doc_batch)} documents...", flush=True)
        all_doc_texts = doc_llm.inference_batch(doc_batch)
        print(f"[Iter {it}/{num_iterations}] Document creation complete.", flush=True)

        # --- Step 4: add new docs to store, re-retrieve with re-ranking ---
        for i, s in enumerate(states):
            new_docs = answers_to_documents([all_doc_texts[i]], iteration=it + 1)
            s.store.add_documents(new_docs)

            chunks, scores = s.store.search_with_scores(s.question_text, k=fetch_k)
            if chunks:
                ai_scores = score_ai_likelihood([c["text"] for c in chunks])
                s.current_docs = rerank_with_ai_penalty(
                    chunks, scores, np.array(ai_scores), lam, search_top_k,
                )
            else:
                s.current_docs = []

    for s in states:
        experiments["questions"].append(s.question_obj)

    write_experiments_output(args.output_path, experiments)
    print(f"\nWrote {args.output_path}")

    try:
        llm.shutdown()
    except Exception:
        pass
    if doc_llm is not llm:
        try:
            doc_llm.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    run_pipeline()
