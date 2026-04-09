"""
Re-ranking formula:
    S_norm = (S1 - S_min) / (S_max - S_min)      # normalise retrieval scores to [0,1]
    S2     = S_norm * (1 - lambda * AI_likelihood) # penalise AI-like docs
"""

import argparse
import gc
import json
import os
import random
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer

from ai_detection_test import load_detector
from formatters import get_create_document_conversation
from pipeline.model_runner import build_llm
from pipeline.output_writer import write_experiments_output
from pipeline.prompt_builder import build_rag_conversation

CACHE_DIR = "/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache/"
INDEX_DIR = "/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpotqa_index/"

DEFAULT_TOP_K = 10
DEFAULT_NPROBE = 64
EMBED_MODEL = "intfloat/e5-small-v2"
EMBED_MAX_LEN = 512
DEFAULT_ROUNDS = 30

class E5Embedder:
    """Thin wrapper around e5-small-v2.  Loads on GPU if available, else CPU."""

    def __init__(self, model_name: str = EMBED_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        print(f"[E5Embedder] Loaded {model_name!r} on {self.device}.")

    @torch.no_grad()
    def encode(self, texts: List[str], prefix: str = "passage: ") -> np.ndarray:
        """Return L2-normalised float32 embeddings. Use prefix='query: ' for queries."""
        prefixed = [f"{prefix}{t}" for t in texts]
        enc = self.tokenizer(
            prefixed, padding=True, truncation=True,
            max_length=EMBED_MAX_LEN, return_tensors="pt",
        ).to(self.device)
        out = self.model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).bool()
        hidden = out.last_hidden_state.masked_fill(~mask, 0.0)
        embs = hidden.sum(dim=1) / enc["attention_mask"].sum(dim=1, keepdim=True)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        return embs.cpu().float().numpy()

class CachedSearchState:
    def __init__(
        self,
        query_vec: np.ndarray,
        corpus_candidates: List[Tuple[float, Dict[str, Any], float]],
    ):
        # corpus_candidates: [(retrieval_score, doc_dict, ai_likelihood), ...]
        self.query_vec = query_vec
        self.corpus_candidates = corpus_candidates
        self.gen_docs: List[Tuple[Dict[str, Any], np.ndarray, float]] = []

    def add_generated_doc(self, doc: Dict[str, Any], vec: np.ndarray, ai_likelihood: float) -> None:
        self.gen_docs.append((doc, vec, ai_likelihood))

    def get_top_k(self, k: int, lam: float) -> List[Dict[str, Any]]:
        # Collect all (raw_score, doc, ai_likelihood)
        candidates: List[Tuple[float, Dict[str, Any], float]] = list(self.corpus_candidates)
        if self.gen_docs:
            gen_vecs = np.stack([v for _, v, _ in self.gen_docs])
            gen_scores = gen_vecs @ self.query_vec
            for (doc, _, ai_lk), s in zip(self.gen_docs, gen_scores):
                candidates.append((float(s), doc, ai_lk))

        if not candidates:
            return []

        # Min-max normalise retrieval scores to [0, 1]
        raw_scores = np.array([s for s, _, _ in candidates], dtype=np.float32)
        s_min, s_max = raw_scores.min(), raw_scores.max()
        if s_max - s_min > 1e-9:
            norm_scores = (raw_scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(raw_scores)

        # Apply penalty: S2 = S_norm * (1 - lambda * AI_likelihood)
        ai_likelihoods = np.array([ai_lk for _, _, ai_lk in candidates], dtype=np.float32)
        penalised = norm_scores * (1.0 - lam * ai_likelihoods)

        # Sort descending by penalised score, deduplicate
        order = np.argsort(-penalised)
        seen: set = set()
        results: List[Dict[str, Any]] = []
        for idx in order:
            doc = candidates[idx][1]
            did = doc.get("doc_id")
            if did in seen:
                continue
            seen.add(did)
            results.append(doc)
            if len(results) >= k:
                break
        return results

def _answers_to_docs(texts: List[str], iteration: int) -> List[Dict[str, Any]]:
    return [
        {"doc_id": f"gen_{iteration}_{i}", "iteration": iteration,
         "url": "model_generated", "text": t}
        for i, t in enumerate(texts)
    ]

def parse_args():
    p = argparse.ArgumentParser(
        description="HotPotQA RAG-collapse pipeline with AI-likelihood re-ranking (search variant)")

    p.add_argument("--vllm-api-base", required=True,
        help="vLLM server base URL for answer generation, e.g. http://host:5150/v1")
    p.add_argument("--model-name", required=True, help="Served model name for answer generation.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--top-p", type=float, default=0.9)

    p.add_argument("--doc-vllm-api-base", required=True,
        help="vLLM server base URL for document generation, e.g. http://host:5151/v1")
    p.add_argument("--doc-model-name", default=None,
        help="Served model name for doc generation. Defaults to --model-name.")
    p.add_argument("--doc-temperature", type=float, default=None,
        help="Temperature for doc generation. Defaults to --temperature.")
    p.add_argument("--doc-max-tokens", type=int, default=None,
        help="Max tokens for doc generation. Defaults to --max-tokens.")
    p.add_argument("--doc-top-p", type=float, default=None,
        help="Top-p for doc generation. Defaults to --top-p.")

    p.add_argument("--split", choices=["train", "test", "val"], default="test")
    p.add_argument("--output-path", default="hotpot_rerank_experiment_output.json")
    p.add_argument("--max-questions", type=int, default=None)

    p.add_argument("--num-iterations", type=int, default=None,
        help=f"Override default round count (default: {DEFAULT_ROUNDS}).")
    p.add_argument("--num-runs", type=int, default=10,
        help="Number of independent runs per iteration.")
    p.add_argument("--chars-per-doc", type=int, default=400,
        help="Character limit per document in the RAG prompt.")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--nprobe", type=int, default=DEFAULT_NPROBE)

    p.add_argument("--lambda-penalty", type=float, default=0.5,
        help="Penalty weight for AI-likelihood. S2 = S1_norm * (1 - lambda * AI_likelihood). "
             "0 = no penalty, 1 = full penalty.")
    p.add_argument("--detector-model", type=str, default="desklib-finetuned",
        help="AI detector model name from MODEL_REGISTRY in ai_detection_test.py.")
    p.add_argument("--detector-batch-size", type=int, default=16,
        help="Batch size for AI detection scoring.")

    p.add_argument("--index-dir", default=INDEX_DIR)
    p.add_argument("--cache-dir", default=CACHE_DIR)

    return p.parse_args()

def run_pipeline() -> None:
    args = parse_args()

    num_iterations = args.num_iterations or DEFAULT_ROUNDS
    num_runs = args.num_runs
    chars_per_doc = args.chars_per_doc
    lam = args.lambda_penalty

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

    queries_path = os.path.join(args.cache_dir, "queries", args.split)
    print(f"Loading queries from {queries_path} ...")
    queries_ds = load_from_disk(queries_path)
    num_questions = (
        len(queries_ds) if args.max_questions is None
        else min(args.max_questions, len(queries_ds))
    )

    print("Loading corpus metadata ...")
    from datasets import load_dataset as hf_load_dataset
    corpus_ds = hf_load_dataset("mteb/hotpotqa", "corpus", cache_dir=args.cache_dir)["corpus"]
    corpus_titles = corpus_ds["title"]
    corpus_texts = corpus_ds["text"]

    print("Loading E5 embedding model ...")
    embedder = E5Embedder()

    print(f"Loading FAISS index from {args.index_dir} ...")
    index = faiss.read_index(os.path.join(args.index_dir, "ivf.index"))
    index.nprobe = args.nprobe
    with open(os.path.join(args.index_dir, "docid_map.json")) as f:
        docid_map: List[str] = json.load(f)

    def _faiss_search(query, top_k):
        """Return (query_vec, [(score, doc_dict), ...]) from FAISS."""
        qvec = embedder.encode([query], prefix="query: ")
        scores, ids = index.search(qvec, top_k)
        candidates = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            did = docid_map[idx]
            candidates.append((float(score), {
                "doc_id": f"corpus_{did}",
                "iteration": 0,
                "url": "",
                "title": corpus_titles[idx],
                "text": corpus_texts[idx],
            }))
        return qvec[0], candidates

    class _QState:
        __slots__ = (
            "q_idx", "query_id", "question_text",
            "current_docs", "search_state", "question_obj",
        )

    print(f"Running initial retrieval for {num_questions} questions ...")
    states: List[_QState] = []
    for q_idx in range(num_questions):
        s = _QState()
        s.q_idx = q_idx
        s.query_id = queries_ds[q_idx]["_id"]
        s.question_text = queries_ds[q_idx]["text"]

        fetch_k = 20
        qvec, candidates = _faiss_search(s.question_text, fetch_k)

        # Score AI-likelihood for all retrieved corpus docs
        doc_texts = [doc["text"] for _, doc in candidates]
        ai_scores = score_ai_likelihood(doc_texts)
        candidates_with_ai = [
            (score, doc, ai_lk)
            for (score, doc), ai_lk in zip(candidates, ai_scores)
        ]

        s.search_state = CachedSearchState(query_vec=qvec, corpus_candidates=candidates_with_ai)
        s.current_docs = s.search_state.get_top_k(args.top_k, lam)

        s.question_obj = {
            "question_id": q_idx,
            "query_id": s.query_id,
            "question_text": s.question_text,
            "iterations": [],
        }
        states.append(s)
        if (q_idx + 1) % 100 == 0:
            print(f"  {q_idx + 1}/{num_questions}")

    print(f"Initial retrieval done for {len(states)} questions.")

    # Free FAISS index and corpus; embedder stays for encoding AI docs each round
    del index, docid_map, corpus_ds, corpus_titles, corpus_texts
    gc.collect()
    print(f"Freed FAISS index and corpus metadata. "
          f"Embedder remains on {embedder.device} for doc encoding.")

    llm, resolved_model_name = build_llm(
        model_mode="server",
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        api_base=args.vllm_api_base,
    )
    print(f"[LLM] Connected. Served model: {getattr(llm, 'served_model_name', resolved_model_name)}", flush=True)

    doc_llm, doc_model_name = build_llm(
        model_mode="server",
        model_name=args.doc_model_name or args.model_name,
        temperature=args.doc_temperature if args.doc_temperature is not None else args.temperature,
        max_tokens=args.doc_max_tokens if args.doc_max_tokens is not None else args.max_tokens,
        top_p=args.doc_top_p if args.doc_top_p is not None else args.top_p,
        api_base=args.doc_vllm_api_base,
    )
    print(f"[Doc LLM] Connected. Served model: {getattr(doc_llm, 'served_model_name', doc_model_name)}", flush=True)

    meta: Dict[str, Any] = {
        "model": resolved_model_name,
        "doc_model": doc_model_name,
        "pipeline_variant": "search_rerank",
        "dataset": "hotpotqa",
        "split": args.split,
        "num_iterations": num_iterations,
        "num_runs_per_iteration": num_runs,
        "top_k": args.top_k,
        "nprobe": args.nprobe,
        "lambda_penalty": lam,
        "detector_model": args.detector_model,
    }
    experiments: Dict[str, Any] = {"experiment_metadata": meta, "questions": []}

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
        print(all_answers[0])
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

        # --- Step 4: score AI-likelihood for new docs, embed, and re-rank ---
        new_texts = [t for t in all_doc_texts]
        ai_scores = score_ai_likelihood(new_texts)

        for i, s in enumerate(states):
            new_doc = _answers_to_docs([all_doc_texts[i]], iteration=it + 1)[0]
            vec = embedder.encode([new_doc["text"]], prefix="passage: ")[0]
            s.search_state.add_generated_doc(new_doc, vec, ai_scores[i])
            s.current_docs = s.search_state.get_top_k(args.top_k, lam)

    for s in states:
        experiments["questions"].append(s.question_obj)

    write_experiments_output(args.output_path, experiments)
    print(f"\nWrote {args.output_path}")

    try:
        llm.shutdown()
    except Exception:
        pass
    try:
        doc_llm.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    run_pipeline()
