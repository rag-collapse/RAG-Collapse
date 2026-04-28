"""
Build the GEPA optimization dataset by combining:
  - 50 random questions from datasets/umass_data.entity.chatgpt.400.jsonl
  - 50 random questions from mteb/hotpotqa, retrieved via FAISS + E5 embeddings

HotpotQA retrieval mirrors hotpot_pipeline.py on the hotpot branch:
  - Corpus : mteb/hotpotqa "corpus" split  (_id, title, text)
  - Queries: mteb/hotpotqa "queries" split (_id, text)
  - Embedder: intfloat/e5-small-v2  (GPU if available, else CPU)
  - Index  : IndexIVFFlat IP on HPC; IndexFlatIP on smoke test
  - Top-k  : 10 documents per question

Smoke-test mode (--smoke-test):
  - Corpus limited to first SMOKE_CORPUS_SIZE rows
  - IndexFlatIP (brute-force, no training needed)
  - Only SMOKE_N questions per source
  - CPU-safe, no HPC infra required

100 combined questions are shuffled and split 60/20/20:
  train.jsonl  — 60 questions  (evaluated each GEPA mutation round)
  val.jsonl    — 20 questions  (ranks candidates at end of each round)
  test.jsonl   — 20 questions  (held out; report final numbers here)

Each JSONL line: {"question": "...", "docs": [{<source fields>, "text": "..."}, ...]}
  UMass docs   : {"url": "https://...", "text": "..."}
  HotpotQA docs: {"doc_id": "corpus_<_id>", "url": "", "title": "...", "text": "..."}

Note: title and doc_id are metadata only; references_to_documents() generates its own
doc_id and only propagates url + text into the live pipeline.

Usage (from repo root):
    # Full HPC run (saves FAISS index for reuse):
    python gepa_optimization/prepare_dataset.py \\
        --cache-dir /scratch/.../hf_cache \\
        --index-dir /scratch/.../hotpotqa_index

    # Smoke test (CPU, ~2 min):
    python gepa_optimization/prepare_dataset.py --smoke-test
"""
import argparse
import gc
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

UMASS_JSONL = REPO_ROOT / "datasets" / "umass_data.entity.chatgpt.400.jsonl"
OUT_DIR     = Path(__file__).parent / "data"
SEED        = 42

UMASS_N  = 50
HOTPOT_N = 50

TRAIN_SIZE = 60
VAL_SIZE   = 20
# remaining 20 go to test

EMBED_MODEL    = "intfloat/e5-small-v2"
EMBED_MAX_LEN  = 512
EMBED_BATCH    = 256
DEFAULT_TOP_K  = 10
IVF_NLIST      = 1024
IVF_NPROBE     = 64

# Smoke-test limits
SMOKE_N           = 10   # questions per source
SMOKE_CORPUS_SIZE = 5000


@dataclass
class RAGDataInst:
    question: str
    docs: list[dict]


# ── E5 Embedder ───────────────────────────────────────────────────────────────

class E5Embedder:
    """
    Thin wrapper around e5-small-v2. Mirrors the E5Embedder in hotpot_pipeline.py.
    Loads on GPU if available, else CPU.
    """

    def __init__(self, model_name: str = EMBED_MODEL, cache_dir: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = (
            AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            .to(self.device)
            .eval()
        )
        print(f"[E5Embedder] Loaded {model_name!r} on {self.device}.")

    def encode(
        self,
        texts: list[str],
        prefix: str = "passage: ",
        batch_size: int = EMBED_BATCH,
    ) -> "np.ndarray":
        """Return L2-normalised float32 embeddings. Use prefix='query: ' for queries."""
        import torch
        import numpy as np
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [f"{prefix}{t}" for t in texts[i : i + batch_size]]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=EMBED_MAX_LEN,
                    return_tensors="pt",
                ).to(self.device)
                out  = self.model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).bool()
                hidden = out.last_hidden_state.masked_fill(~mask, 0.0)
                embs = hidden.sum(dim=1) / enc["attention_mask"].sum(dim=1, keepdim=True)
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                all_embs.append(embs.cpu().float().numpy())
        return np.vstack(all_embs)


# ── FAISS helpers ─────────────────────────────────────────────────────────────

def _build_flat_index(vecs):
    """Brute-force inner-product index. CPU-safe, no training. For smoke test."""
    import faiss
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def _build_ivf_index(vecs, nlist: int = IVF_NLIST):
    """IVF flat index with IP metric. Matches hotpot_pipeline.py for HPC runs."""
    import faiss
    d = vecs.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)
    index.add(vecs)
    return index


# ── UMass dataset ─────────────────────────────────────────────────────────────

def load_umass(jsonl_path: Path = UMASS_JSONL, n: int = UMASS_N) -> list[RAGDataInst]:
    """Load n random questions from the UMass JSONL dataset."""
    with open(jsonl_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    random.shuffle(rows)
    rows = rows[:n]
    return [
        RAGDataInst(question=row["question"], docs=row["references"])
        for row in rows
    ]


# ── HotpotQA dataset ──────────────────────────────────────────────────────────

def load_hotpot(
    n: int = HOTPOT_N,
    cache_dir: str | None = None,
    index_dir: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    smoke_test: bool = False,
) -> list[RAGDataInst]:
    """
    Load n questions from mteb/hotpotqa with top-k retrieved docs each, using
    E5 + FAISS — mirroring hotpot_pipeline.py on the hotpot branch.

    smoke_test=True: corpus capped at SMOKE_CORPUS_SIZE, IndexFlatIP (no training).
    index_dir: if provided and index exists, reuse it; otherwise build + save.
    """
    from datasets import load_dataset

    # ── Load corpus ──────────────────────────────────────────────────────────
    corpus_limit = SMOKE_CORPUS_SIZE if smoke_test else None
    corpus_split = f"corpus[:{corpus_limit}]" if corpus_limit else "corpus"
    tag = f"(smoke: first {corpus_limit} rows)" if smoke_test else "(full)"
    print(f"Loading mteb/hotpotqa corpus {tag} ...")
    corpus_ds     = load_dataset("mteb/hotpotqa", "corpus", split=corpus_split, cache_dir=cache_dir)
    corpus_ids    = corpus_ds["_id"]
    corpus_titles = corpus_ds["title"]
    corpus_texts  = corpus_ds["text"]
    print(f"  Corpus: {len(corpus_ids):,} passages")

    # ── Build or load FAISS index ─────────────────────────────────────────────
    index_path  = Path(index_dir) / "ivf.index"      if index_dir else None
    docmap_path = Path(index_dir) / "docid_map.json" if index_dir else None

    if not smoke_test and index_path and index_path.exists():
        import faiss
        print(f"Loading existing FAISS index from {index_dir} ...")
        index = faiss.read_index(str(index_path))
        with open(docmap_path) as f:
            docid_map: list[str] = json.load(f)
    else:
        print("Encoding corpus with E5 ...")
        embedder = E5Embedder(cache_dir=cache_dir)
        corpus_vecs = embedder.encode(corpus_texts, prefix="passage: ")
        del embedder
        gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except ImportError:
            pass

        if smoke_test:
            print("Building IndexFlatIP (smoke test, brute-force) ...")
            index = _build_flat_index(corpus_vecs)
        else:
            print(f"Building IndexIVFFlat (nlist={IVF_NLIST}) ...")
            index = _build_ivf_index(corpus_vecs)
            if index_dir:
                import faiss
                Path(index_dir).mkdir(parents=True, exist_ok=True)
                faiss.write_index(index, str(index_path))
                with open(docmap_path, "w") as f:
                    json.dump(corpus_ids, f)
                print(f"Saved FAISS index -> {index_dir}")

        docid_map = list(corpus_ids)

    if not smoke_test:
        index.nprobe = IVF_NPROBE  # applies whether index was loaded or just built

    # ── Load queries and retrieve ─────────────────────────────────────────────
    print("Loading mteb/hotpotqa queries ...")
    queries_ds    = load_dataset("mteb/hotpotqa", "queries", split="queries", cache_dir=cache_dir)
    all_questions = [(row["_id"], row["text"]) for row in queries_ds]
    random.shuffle(all_questions)
    selected = all_questions[:n]
    print(f"  Selected {len(selected)} questions")

    print(f"Encoding {len(selected)} queries with E5 ...")
    embedder = E5Embedder(cache_dir=cache_dir)
    q_vecs   = embedder.encode([q for _, q in selected], prefix="query: ")
    del embedder
    gc.collect()

    scores_mat, ids_mat = index.search(q_vecs, top_k)

    instances = []
    for i, (qid, question) in enumerate(selected):
        docs = []
        for idx in ids_mat[i]:
            if idx < 0:
                continue
            docs.append({
                "doc_id": f"corpus_{docid_map[idx]}",
                "url":    "",
                "title":  corpus_titles[idx],
                "text":   corpus_texts[idx],
            })
        if docs:
            instances.append(RAGDataInst(question=question, docs=docs))

    return instances


# ── Combined split ────────────────────────────────────────────────────────────

def prepare(
    out_dir: Path = OUT_DIR,
    seed: int = SEED,
    cache_dir: str | None = None,
    index_dir: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    smoke_test: bool = False,
):
    random.seed(seed)

    umass_n  = SMOKE_N  if smoke_test else UMASS_N
    hotpot_n = SMOKE_N  if smoke_test else HOTPOT_N

    print(f"Loading {umass_n} UMass questions...")
    umass = load_umass(n=umass_n)
    print(f"  Loaded {len(umass)} UMass instances.")

    print(f"Loading {hotpot_n} HotpotQA questions...")
    hotpot = load_hotpot(
        n=hotpot_n,
        cache_dir=cache_dir,
        index_dir=index_dir,
        top_k=top_k,
        smoke_test=smoke_test,
    )
    print(f"  Loaded {len(hotpot)} HotpotQA instances.")

    combined = umass + hotpot
    random.shuffle(combined)

    if smoke_test:
        n = len(combined)
        t = max(1, int(n * 0.6))
        v = max(1, int(n * 0.2))
        splits = {
            "train": combined[:t],
            "val":   combined[t : t + v],
            "test":  combined[t + v :],
        }
    else:
        splits = {
            "train": combined[:TRAIN_SIZE],
            "val":   combined[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE],
            "test":  combined[TRAIN_SIZE + VAL_SIZE :],
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for inst in data:
                f.write(json.dumps({"question": inst.question, "docs": inst.docs}) + "\n")
        print(f"Saved {len(data):>3} instances -> {path}")

    total = sum(len(d) for d in splits.values())
    print(
        f"\nTotal: {total} | Train: {len(splits['train'])} "
        f"| Val: {len(splits['val'])} | Test: {len(splits['test'])}"
        + ("  (smoke test)" if smoke_test else "")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build GEPA optimization dataset")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help=(
            f"Quick CPU-safe run: corpus capped at {SMOKE_CORPUS_SIZE} rows, "
            f"IndexFlatIP (no training), {SMOKE_N} questions per source."
        ),
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--index-dir", default=None,
        help="Directory to save/load FAISS index. Reused if it already exists.",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Documents retrieved per question (default: {DEFAULT_TOP_K})",
    )
    args = parser.parse_args()
    prepare(
        cache_dir=args.cache_dir,
        index_dir=args.index_dir,
        top_k=args.top_k,
        smoke_test=args.smoke_test,
    )
