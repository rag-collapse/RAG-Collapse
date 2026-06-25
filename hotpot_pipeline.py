"""
Memory strategy
Initial retrieval:  E5 embedder lives on GPU (falls back to CPU if none).
                    Load FAISS index + corpus metadata, do all initial
                    retrievals, then free the index and corpus.
                    For the search variant the embedder stays alive on GPU
                    to encode AI-generated docs each round.
                    For hybrid / replace_one the embedder is deleted after
                    initial retrieval — it is no longer needed.
Main loop:          Calls remote vLLM servers for answer and doc generation.
                    Search variant encodes each new doc on GPU and merges
                    scores with the cached FAISS candidates (numpy dot product).
"""

import argparse
import copy
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

from formatters import get_create_document_conversation
from pipeline.misinfo import (
    MisinfoController, MODE_FAITHFUL, MODES, TARGETS,
    DistractorController, DISTRACTOR_MODES,
    load_native_records, native_context_docs,
    per_run_doc_subsets,
)
from pipeline.model_runner import build_llm
from pipeline.output_writer import write_experiments_output
from pipeline.prompt_builder import build_rag_conversation

CACHE_DIR = "/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache/"
INDEX_DIR = "/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpotqa_index/"

DEFAULT_TOP_K = 10
DEFAULT_NPROBE = 64
EMBED_MODEL = "intfloat/e5-small-v2"
EMBED_MAX_LEN = 512

VARIANT_SEARCH = "search"
VARIANT_HYBRID = "hybrid"
VARIANT_REPLACE_ONE = "replace_one"
ALL_VARIANTS = (VARIANT_SEARCH, VARIANT_HYBRID, VARIANT_REPLACE_ONE)

DEFAULT_ROUNDS = {
    VARIANT_SEARCH: 30,
    VARIANT_HYBRID: 10,
    VARIANT_REPLACE_ONE: 20,
}
MAX_INITIAL_DOCS_REPLACE_ONE = 10

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
        corpus_candidates: List[Tuple[float, Dict[str, Any]]],
    ):
        self.query_vec = query_vec                      # (dim,)
        self.corpus_candidates = corpus_candidates      # [(score, doc), ...]
        self.gen_docs: List[Tuple[Dict[str, Any], np.ndarray]] = []

    def add_generated_doc(self, doc: Dict[str, Any], vec: np.ndarray) -> None:
        self.gen_docs.append((doc, vec))

    def get_top_k(self, k: int) -> List[Dict[str, Any]]:
        candidates = list(self.corpus_candidates)
        if self.gen_docs:
            gen_vecs = np.stack([v for _, v in self.gen_docs])
            gen_scores = gen_vecs @ self.query_vec
            for (doc, _), s in zip(self.gen_docs, gen_scores):
                candidates.append((float(s), doc))
        candidates.sort(key=lambda x: x[0], reverse=True)
        seen: set = set()
        results: List[Dict[str, Any]] = []
        for _, doc in candidates:
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


def _select(items: list, k: int, mode: str) -> list:
    if k <= 0:
        return []
    if mode == "first" or k >= len(items):
        return list(items[:k])
    return list(random.sample(items, k))


def _next_docs_hybrid(
    doc_texts: List[str],
    initial_corpus_docs: List[Dict[str, Any]],
    iteration: int,
    num_synth: int,
    num_db: int,
    synth_selection: str,
    db_selection: str,
) -> List[Dict[str, Any]]:
    selected_texts = _select(doc_texts, num_synth, synth_selection)
    synth_docs = _answers_to_docs(selected_texts, iteration) if selected_texts else []
    db_docs = _select(initial_corpus_docs, num_db, db_selection)
    return synth_docs + db_docs


def _next_docs_replace_one(
    current_docs: List[Dict[str, Any]],
    new_doc_texts: List[str],
    iteration: int,
) -> List[Dict[str, Any]]:
    if not new_doc_texts or not current_docs:
        return current_docs
    slot = iteration % len(current_docs)
    next_docs = copy.deepcopy(current_docs)
    next_docs[slot] = _answers_to_docs([new_doc_texts[0]], iteration)[0]
    return next_docs

def parse_args():
    p = argparse.ArgumentParser(description="HotPotQA RAG-collapse pipeline")

    p.add_argument("--vllm-api-base", required=True,
        help="vLLM server base URL for answer generation, e.g. http://host:5150/v1")
    p.add_argument("--model-name", required=True, help="Served model name for answer generation.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--top-p", type=float, default=0.9)

    p.add_argument("--doc-model-mode", choices=["api", "local", "server"], default="server",
        help="Backend for document/distractor generation. 'server' (default) = vLLM HTTP "
             "(needs --doc-vllm-api-base); 'api' = keymaker LiteLLM strong model (needs API_KEY); "
             "'local' = in-process vLLM (GPU).")
    p.add_argument("--doc-vllm-api-base", default=None,
        help="vLLM server base URL for document generation, e.g. http://host:5151/v1. "
             "Required when --doc-model-mode=server.")
    p.add_argument("--doc-model-name", default=None,
        help="Model name for doc generation. Defaults to --model-name. For --doc-model-mode=api "
             "use a keymaker id, e.g. openai/claude-sonnet-4-6.")
    p.add_argument("--doc-temperature", type=float, default=None,
        help="Temperature for doc generation. Defaults to --temperature.")
    p.add_argument("--doc-max-tokens", type=int, default=None,
        help="Max tokens for doc generation. Defaults to --max-tokens.")
    p.add_argument("--doc-top-p", type=float, default=None,
        help="Top-p for doc generation. Defaults to --top-p.")

    p.add_argument("--split", choices=["train", "test", "val"], default="test")
    p.add_argument("--output-path", default="hotpot_experiment_output.json")
    p.add_argument("--max-questions", type=int, default=None)

    p.add_argument("--pipeline-variant", choices=list(ALL_VARIANTS), default="search")
    p.add_argument("--num-iterations", type=int, default=None,
        help="Override default round count for the variant.")
    p.add_argument("--num-runs", type=int, default=10,
        help="Number of independent runs per iteration.")
    p.add_argument("--chars-per-doc", type=int, default=400,
        help="Character limit per document in the RAG prompt.")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--nprobe", type=int, default=DEFAULT_NPROBE)

    p.add_argument("--num-synth-docs", type=int, default=1)
    p.add_argument("--num-db-docs", type=int, default=3,
        help="Corpus docs per round. 0 = replace-all style.")
    p.add_argument("--db-doc-selection", choices=["first", "random"], default="first")
    p.add_argument("--synth-doc-selection", choices=["first", "random"], default="first")

    p.add_argument("--index-dir", default=INDEX_DIR)
    p.add_argument("--cache-dir", default=CACHE_DIR)

    # --- Misinformation injection (error-compounding experiment; all default-off) ---
    p.add_argument("--doc-synthesis-mode", choices=list(MODES), default=MODE_FAITHFUL,
        help="faithful (baseline), counterfactual (entity substitution), or freeform "
             "(synthesizer invents one false claim). Non-faithful requires --gt-file.")
    p.add_argument("--target-mode", choices=list(TARGETS), default="final_answer",
        help="What to corrupt: the final answer entity, an intermediate bridge entity "
             "(requires --native-hotpot-file), or an answer-irrelevant detail (untargeted control).")
    p.add_argument("--inject-round", type=int, default=1,
        help="Iteration index whose synthesized document is corrupted; that doc enters "
             "the context/corpus at iteration inject_round+1.")
    p.add_argument("--inject-every-round", action="store_true",
        help="Stress arm: corrupt the synthesized document every round (not just --inject-round).")
    p.add_argument("--seed", type=int, default=None,
        help="Global seed (random, numpy). Substitute selection uses a separate per-question "
             "RNG so control/treatment arms select identical answers under the same seed.")
    p.add_argument("--gt-file", default=None,
        help="HotpotQA ground-truth JSON (native list of {_id, answer,...} or flat {id: answer}). "
             "Required when --doc-synthesis-mode != faithful.")
    p.add_argument("--native-hotpot-file", default=None,
        help="Native HotpotQA JSON with supporting_facts/context/type. Required for "
             "--target-mode intermediate_hop.")

    # --- Round-0 initial-document distractors (widen the starting distribution; default-off) ---
    # Independent of --doc-synthesis-mode: corrupts a fraction of each question's round-0
    # retrieved docs into wrong-answer distractors, each carrying its OWN distinct falsehood
    # (DIVERSE) to simulate the spread of independently-hallucinated AI documents.
    p.add_argument("--distractor-fraction", type=float, default=0.0,
        help="Fraction of round-0 retrieved docs to turn into wrong-answer distractors "
             "(0 = off). Requires --gt-file.")
    p.add_argument("--distractor-mode", choices=list(DISTRACTOR_MODES), default="rewrite",
        help="How to build each distractor: rewrite (doc-LLM invents a distinct wrong answer "
             "per doc), substitution (distinct same-type wrong entity per doc), diverse_synth "
             "(coordinated distinct wrong answers, one Wikipedia-style doc each), equal_diverse_synth "
             "(fewer wrong answers, each reinforced by --distractor-docs-per-topic paragraphs; driven "
             "by --distractor-num-topics, not --distractor-fraction), or native_noise "
             "(real non-answer HotpotQA paragraphs; requires --native-hotpot-file).")
    p.add_argument("--distractor-num-topics", type=int, default=0,
        help="equal_diverse_synth only: number of distinct wrong-answer TOPICS to seed (0 = off). Each "
             "topic gets --distractor-docs-per-topic paragraphs, so this drives the arm in place of "
             "--distractor-fraction. Clamped by the available non-gold slots.")
    p.add_argument("--distractor-docs-per-topic", type=int, default=2,
        help="equal_diverse_synth only: paragraphs per topic (distinct generations asserting the SAME "
             "wrong answer). Default 2 → 4 topics fills the 8 non-gold slots of the native setting.")
    p.add_argument("--distractor-per-run", action="store_true",
        help="Option A: give each of the --num-runs runs a DIFFERENT single distractor doc "
             "(clean/gold docs + one rotating distractor) instead of all distractors at once, so "
             "the round-0 answer distribution is wide across runs. No-op without distractor docs.")
    p.add_argument("--distractor-avoid-gold", action="store_true",
        help="Never corrupt a gold document (tagged gold=True or containing the gold answer); "
             "randomly inject distractors into the NON-gold docs only. Use with --initial-docs "
             "native_distractor to keep the 2 gold paragraphs intact and corrupt the 8 distractors.")
    # Separate model for ROUND-0 distractor generation only (two-server style): a strong model
    # seeds the distractors, while the ANSWER model (--doc-model-*) builds the per-round AI docs.
    p.add_argument("--distractor-model-mode", choices=["api", "local", "server"], default=None,
        help="Backend for round-0 distractor generation. Defaults to the --doc-model-* backend "
             "when unset. Use 'api' with a strong keymaker model (e.g. azure/gpt-5-mini) to seed "
             "distractors while per-round synthesis stays on the answer model.")
    p.add_argument("--distractor-model-name", default=None,
        help="Model for round-0 distractor generation (e.g. azure/gpt-5-mini). Defaults to the "
             "doc model when unset.")
    p.add_argument("--distractor-vllm-api-base", default=None,
        help="vLLM base URL for the distractor model when --distractor-model-mode=server.")
    p.add_argument("--distractor-max-tokens", type=int, default=None,
        help="Max tokens for the distractor model. Defaults to --doc-max-tokens. Set higher (e.g. "
             "2048) for reasoning distractor models (gpt-5*) WITHOUT inflating the per-round doc budget.")

    # --- Original HotpotQA paper distractor setting (round-0 source; default-off) ---
    # Replicate Yang et al. (EMNLP 2018): seed round 0 from each question's native
    # 2-gold + 8-TF-IDF-distractor context instead of FAISS retrieval. Distinct from the
    # synthetic --distractor-fraction above (those assert a wrong answer; the native ones
    # are answer-absent hard negatives).
    p.add_argument("--initial-docs", choices=["faiss", "native_distractor"], default="faiss",
        help="Round-0 document source. 'faiss' (default): retrieve from the Wikipedia FAISS "
             "index (pipeline unchanged). 'native_distractor': seed round 0 from the question's "
             "native HotpotQA distractor-setting context (2 gold + 8 TF-IDF distractors); "
             "requires --native-hotpot-file (hotpot_dev_distractor_v1.json).")
    p.add_argument("--distractor-gold-only", action="store_true",
        help="With --initial-docs native_distractor: keep ONLY the 2 gold paragraphs (drop the 8 "
             "distractors) — the no-distractor contrast for the distractor-setting experiment.")

    return p.parse_args()


def run_pipeline() -> None:
    args = parse_args()

    # Reproducibility: seed the global RNGs (answer selection, doc shuffling).
    # Substitute selection in MisinfoController uses a SEPARATE per-question RNG so
    # the global sequence — and hence the answers selected — is identical across arms.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"[seed] global RNG seeded with {args.seed}")

    if args.doc_model_mode == "server" and not args.doc_vllm_api_base:
        raise SystemExit("--doc-vllm-api-base is required when --doc-model-mode=server")

    inject_enabled = args.doc_synthesis_mode != MODE_FAITHFUL
    if inject_enabled and not args.gt_file:
        raise SystemExit("--gt-file is required when --doc-synthesis-mode != faithful")
    if args.target_mode == "intermediate_hop" and not args.native_hotpot_file:
        raise SystemExit("--native-hotpot-file is required when --target-mode intermediate_hop")

    # equal_diverse_synth is driven by --distractor-num-topics; all other modes by --distractor-fraction.
    equal_mode = args.distractor_mode == "equal_diverse_synth"
    distractor_enabled = (
        (args.distractor_num_topics and args.distractor_num_topics > 0) if equal_mode
        else (args.distractor_fraction and args.distractor_fraction > 0)
    )
    if distractor_enabled and not args.gt_file:
        raise SystemExit("--gt-file is required when distractors are enabled")
    if distractor_enabled and args.distractor_mode == "native_noise" and not args.native_hotpot_file:
        raise SystemExit("--native-hotpot-file is required when --distractor-mode native_noise")

    native_seed = args.initial_docs == "native_distractor"
    if native_seed and not args.native_hotpot_file:
        raise SystemExit("--initial-docs native_distractor requires --native-hotpot-file "
                         "(the hotpot_dev_distractor_v1.json distractor-setting file)")
    # native_distractor + synthetic distractors are composable: seed the native 2-gold/8-distractor
    # context, then convert a fraction of the NON-gold slots into wrong-answer docs. Pair with
    # --distractor-avoid-gold so the 2 gold paragraphs are never corrupted.
    if native_seed and distractor_enabled and not args.distractor_avoid_gold:
        print("[warn] --initial-docs native_distractor with synthetic distractors but WITHOUT "
              "--distractor-avoid-gold: gold paragraphs may be corrupted.", flush=True)

    variant = args.pipeline_variant
    num_iterations = args.num_iterations or DEFAULT_ROUNDS[variant]
    num_runs = args.num_runs
    chars_per_doc = args.chars_per_doc
    is_search = variant == VARIANT_SEARCH

    queries_path = os.path.join(args.cache_dir, "queries", args.split)
    print(f"Loading queries from {queries_path} ...")
    queries_ds = load_from_disk(queries_path)
    num_questions = (
        len(queries_ds) if args.max_questions is None
        else min(args.max_questions, len(queries_ds))
    )

    if native_seed:
        # Original-HotpotQA distractor setting: no FAISS / Wikipedia corpus — round-0 docs come
        # from each question's native context. Embedder only needed for the search variant.
        print(f"Loading native HotpotQA distractor-setting file {args.native_hotpot_file} ...")
        native_records = load_native_records(args.native_hotpot_file)
        index = docid_map = corpus_ds = corpus_titles = corpus_texts = None
        embedder = E5Embedder() if is_search else None
    else:
        print("Loading corpus metadata ...")
        from datasets import load_dataset as hf_load_dataset
        corpus_ds = hf_load_dataset("mteb/hotpotqa", "corpus", cache_dir=args.cache_dir)["corpus"]
        corpus_titles = corpus_ds["title"]
        corpus_texts = corpus_ds["text"]

        print("Loading E5 embedding model ...")
        embedder = E5Embedder()  # GPU if available, else CPU

        print(f"Loading FAISS index from {args.index_dir} ...")
        index = faiss.read_index(os.path.join(args.index_dir, "ivf.index"))
        index.nprobe = args.nprobe
        with open(os.path.join(args.index_dir, "docid_map.json")) as f:
            docid_map: List[str] = json.load(f)

    def _faiss_search(query, top_k):
        """Return (query_vec, [(score, doc_dict), ...]) from FAISS."""
        qvec = embedder.encode([query], prefix="query: ")  # (1, dim)
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
            "current_docs", "initial_corpus_docs",
            "search_state", "question_obj",
        )

    print(f"Running initial retrieval for {num_questions} questions ...")
    states: List[_QState] = []
    n_missing = 0
    for q_idx in range(num_questions):
        s = _QState()
        s.q_idx = q_idx
        s.query_id = queries_ds[q_idx]["_id"]
        s.question_text = queries_ds[q_idx]["text"]
        s.search_state = None

        if native_seed:
            # Round-0 = the question's native distractor-setting context (2 gold + 8 distractors).
            rec = native_records.get(s.query_id)
            if rec is None:
                n_missing += 1
                continue
            docs = native_context_docs(rec, gold_only=args.distractor_gold_only)
            s.initial_corpus_docs = docs
            s.current_docs = list(docs)
            if variant == VARIANT_HYBRID and args.num_db_docs > 0:
                s.current_docs = _select(docs, args.num_db_docs, args.db_doc_selection)
            if is_search:
                # Search universe = the native paragraphs (embedded), growing with generated docs.
                qv = embedder.encode([s.question_text], prefix="query: ")[0]
                if docs:
                    dvecs = embedder.encode([d["text"] for d in docs], prefix="passage: ")
                    candidates = [(float(sc), d) for sc, d in zip(dvecs @ qv, docs)]
                else:
                    candidates = []
                s.search_state = CachedSearchState(query_vec=qv, corpus_candidates=candidates)
        else:
            k = MAX_INITIAL_DOCS_REPLACE_ONE if variant == VARIANT_REPLACE_ONE else args.top_k
            fetch_k = k + 20 if is_search else k
            qvec, candidates = _faiss_search(s.question_text, fetch_k)

            s.initial_corpus_docs = [doc for _, doc in candidates[:k]]
            s.current_docs = list(s.initial_corpus_docs)

            if variant == VARIANT_HYBRID and args.num_db_docs > 0:
                s.current_docs = _select(s.initial_corpus_docs, args.num_db_docs, args.db_doc_selection)

            if is_search:
                s.search_state = CachedSearchState(query_vec=qvec, corpus_candidates=candidates)

        s.question_obj = {
            "question_id": q_idx,
            "query_id": s.query_id,
            "question_text": s.question_text,
            "iterations": [],
        }
        states.append(s)
        if (q_idx + 1) % 100 == 0:
            print(f"  {q_idx + 1}/{num_questions}")

    if native_seed and n_missing:
        print(f"[native] {n_missing}/{num_questions} questions not found in the distractor file (skipped).")
    if not states:
        raise SystemExit("No questions to run — check that --native-hotpot-file (distractor setting) "
                         "aligns with the query --split.")
    print(f"Initial retrieval done for {len(states)} questions.")

    # Free FAISS index and corpus (faiss path only; the native path never loaded them).
    # For search: embedder stays on GPU to encode AI-generated docs each round.
    # For hybrid / replace_one: embedder is no longer needed.
    if not native_seed:
        del index, docid_map, corpus_ds, corpus_titles, corpus_texts
    gc.collect()

    if not is_search:
        if embedder is not None:
            del embedder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Freed embedder, FAISS index, and corpus metadata.")
    else:
        print(f"Freed FAISS index and corpus metadata. "
              f"Embedder remains on {embedder.device} for search-variant doc encoding.")

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
        model_mode=args.doc_model_mode,
        model_name=args.doc_model_name or args.model_name,
        temperature=args.doc_temperature if args.doc_temperature is not None else args.temperature,
        max_tokens=args.doc_max_tokens if args.doc_max_tokens is not None else args.max_tokens,
        top_p=args.doc_top_p if args.doc_top_p is not None else args.top_p,
        api_base=args.doc_vllm_api_base,
    )
    print(f"[Doc LLM] Connected. Served model: {getattr(doc_llm, 'served_model_name', doc_model_name)}", flush=True)

    controller = None
    if inject_enabled:
        controller = MisinfoController(
            mode=args.doc_synthesis_mode,
            target_mode=args.target_mode,
            inject_round=args.inject_round,
            inject_every_round=args.inject_every_round,
            gt_file=args.gt_file,
            doc_llm=doc_llm,
            seed=args.seed,
            native_file=args.native_hotpot_file,
        )
        print(f"[misinfo] mode={args.doc_synthesis_mode} target={args.target_mode} "
              f"inject_round={args.inject_round} every={args.inject_every_round}", flush=True)
        controller.prepare(states)
        n_elig = sum(1 for s in states if controller.records[s.query_id].eligible)
        print(f"[misinfo] prepared {len(states)} questions; {n_elig} eligible for injection.", flush=True)

    # Round-0 distractors: corrupt a fraction of each question's initial docs IN PLACE,
    # before the loop reads them. Independent of (and composable with) the controller above.
    # The distractor model is SEPARATE from doc_llm (two-server style): a strong model seeds the
    # round-0 distractors, while doc_llm (the answer model) builds the per-round AI docs.
    distractor_llm = doc_llm
    if distractor_enabled and args.distractor_model_name:
        _dist_max_tokens = (
            args.distractor_max_tokens if args.distractor_max_tokens is not None
            else (args.doc_max_tokens if args.doc_max_tokens is not None else args.max_tokens)
        )
        distractor_llm, distractor_model_name = build_llm(
            model_mode=args.distractor_model_mode or "api",
            model_name=args.distractor_model_name,
            temperature=args.doc_temperature if args.doc_temperature is not None else args.temperature,
            max_tokens=_dist_max_tokens,
            top_p=args.doc_top_p if args.doc_top_p is not None else args.top_p,
            api_base=args.distractor_vllm_api_base,
        )
        print(f"[Distractor LLM] Connected. Served model: "
              f"{getattr(distractor_llm, 'served_model_name', distractor_model_name)}", flush=True)

    distractor_controller = None
    if distractor_enabled:
        distractor_controller = DistractorController(
            mode=args.distractor_mode,
            fraction=args.distractor_fraction,
            gt_file=args.gt_file,
            doc_llm=distractor_llm,
            seed=args.seed,
            native_file=args.native_hotpot_file,
            avoid_gold=args.distractor_avoid_gold,
            num_topics=args.distractor_num_topics,
            docs_per_topic=args.distractor_docs_per_topic,
        )
        if equal_mode:
            print(f"[distractor] mode={args.distractor_mode} "
                  f"num_topics={args.distractor_num_topics} docs_per_topic={args.distractor_docs_per_topic}",
                  flush=True)
        else:
            print(f"[distractor] mode={args.distractor_mode} fraction={args.distractor_fraction}", flush=True)
        distractor_controller.prepare_and_apply(states)
        d_elig = sum(1 for s in states if distractor_controller.records[s.query_id].eligible)
        d_docs = sum(distractor_controller.records[s.query_id].n_corrupted for s in states)
        print(f"[distractor] {d_elig}/{len(states)} questions corrupted; {d_docs} distractor docs.", flush=True)

    meta: Dict[str, Any] = {
        "model": resolved_model_name,
        "doc_model": doc_model_name,
        "pipeline_variant": variant,
        "dataset": "hotpotqa",
        "split": args.split,
        "num_iterations": num_iterations,
        "num_runs_per_iteration": num_runs,
        "top_k": args.top_k,
        "nprobe": args.nprobe,
    }
    if native_seed:
        meta["initial_docs"] = args.initial_docs
        meta["native_hotpot_file"] = args.native_hotpot_file
        meta["distractor_gold_only"] = args.distractor_gold_only
    if variant == VARIANT_HYBRID:
        meta["num_synth_docs"] = args.num_synth_docs
        meta["num_db_docs"] = args.num_db_docs
        meta["db_doc_selection"] = args.db_doc_selection
        meta["synth_doc_selection"] = args.synth_doc_selection
    if controller is not None:
        meta.update(controller.metadata())
        meta["gt_file"] = args.gt_file
        meta["native_hotpot_file"] = args.native_hotpot_file
    if distractor_controller is not None:
        meta.update(distractor_controller.metadata())
        meta["gt_file"] = args.gt_file

    experiments: Dict[str, Any] = {"experiment_metadata": meta, "questions": []}

    for it in range(num_iterations):
        print(f"\n=== Iteration {it}/{num_iterations} ({variant}) ===")

        # --- Step 1: batch answer generation ---
        batch_conversations: List[List[Dict[str, str]]] = []
        for s in states:
            if args.distractor_per_run:
                # Option A: each run sees clean/gold docs + ONE rotating distractor → diverse
                # round-0 answers across runs (instead of all runs sharing one context).
                for docs_r in per_run_doc_subsets(s.current_docs, num_runs):
                    batch_conversations.append(build_rag_conversation(
                        question=s.question_text,
                        docs=docs_r,
                        chars_per_doc=chars_per_doc,
                        shuffle_docs=True,
                    ))
            else:
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
        docs_per_q: List[int] = []
        for i, s in enumerate(states):
            answers = per_q_answers[i]
            if variant == VARIANT_HYBRID:
                ans_for_docs = (
                    answers[:args.num_synth_docs]
                    if args.synth_doc_selection == "first"
                    else random.sample(answers, min(args.num_synth_docs, len(answers)))
                )
            else:
                ans_for_docs = [random.choice(answers)]
            for a in ans_for_docs:
                if controller is not None:
                    doc_batch.append(controller.make_doc_conversation(s, a, it))
                else:
                    doc_batch.append(get_create_document_conversation(
                        question=s.question_text, answer=a,
                    ))
            docs_per_q.append(len(ans_for_docs))

        print(f"[Iter {it}/{num_iterations}] Creating {len(doc_batch)} documents...", flush=True)
        all_doc_texts = doc_llm.inference_batch(doc_batch)
        print(f"[Iter {it}/{num_iterations}] Document creation complete.", flush=True)

        # --- Step 4: update docs for next round ---
        offset = 0
        synth_records: List[Tuple[Any, List[str], List[str]]] = []
        for i, s in enumerate(states):
            n = docs_per_q[i]
            doc_texts = all_doc_texts[offset:offset + n]
            offset += n
            synth_records.append(
                (s, doc_texts, [f"gen_{it + 1}_{j}" for j in range(n)])
            )

            if is_search:
                new_doc = _answers_to_docs([doc_texts[0]], iteration=it + 1)[0]
                vec = embedder.encode([new_doc["text"]], prefix="passage: ")[0]
                s.search_state.add_generated_doc(new_doc, vec)
                s.current_docs = s.search_state.get_top_k(args.top_k)

            elif variant == VARIANT_REPLACE_ONE:
                s.current_docs = _next_docs_replace_one(
                    s.current_docs, doc_texts, iteration=it + 1,
                )

            else:  # hybrid
                s.current_docs = _next_docs_hybrid(
                    doc_texts,
                    s.initial_corpus_docs,
                    iteration=it + 1,
                    num_synth=args.num_synth_docs,
                    num_db=args.num_db_docs,
                    synth_selection=args.synth_doc_selection,
                    db_selection=args.db_doc_selection,
                )

        if controller is not None:
            controller.record_injection(synth_records, it)

    if controller is not None:
        controller.attach(states)
    if distractor_controller is not None:
        distractor_controller.attach(states)

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
    if distractor_llm is not doc_llm:
        try:
            distractor_llm.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    run_pipeline()
