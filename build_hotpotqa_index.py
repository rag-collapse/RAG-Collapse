import os
import numpy as np
import faiss
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import json

# ── paths ──────────────────────────────────────────────────────────────────────
cache_dir   = "/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hf_cache/"
index_dir   = "/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hotpotqa_index/"
os.makedirs(index_dir, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────────
MODEL_NAME  = "intfloat/e5-small-v2"
BATCH_SIZE  = 512
MAX_LENGTH  = 512
NLIST       = 4096   # IVF clusters; ~sqrt(N) for N≈5M docs
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── load corpus ────────────────────────────────────────────────────────────────
print("Loading corpus...")
corpus_ds = load_dataset("mteb/hotpotqa", "corpus", cache_dir=cache_dir)["corpus"]
# fields: _id, title, text

ids    = corpus_ds["_id"]
titles = corpus_ds["title"]
texts  = corpus_ds["text"]

# save id -> int mapping so we can map FAISS results back to doc ids
id2int = {doc_id: i for i, doc_id in enumerate(ids)}
with open(os.path.join(index_dir, "docid_map.json"), "w") as f:
    json.dump(list(ids), f)   # list; position = faiss int id

# ── load model ─────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class CorpusDataset(Dataset):
    def __init__(self, titles, texts):
        self.titles = titles
        self.texts  = texts
    def __len__(self):
        return len(self.titles)
    def __getitem__(self, i):
        return f"passage: {self.titles[i]} {self.texts[i]}"

def collate_fn(batch_texts):
    return tokenizer(batch_texts, padding=True, truncation=True,
                     max_length=MAX_LENGTH, return_tensors="pt")

# ── encode corpus ──────────────────────────────────────────────────────────────
n_docs  = len(ids)
dim     = 384   # e5-small-v2 hidden size
dataset = CorpusDataset(titles, texts)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4,
                     collate_fn=collate_fn, pin_memory=(DEVICE == "cuda"))

print(f"Encoding {n_docs:,} documents in batches of {BATCH_SIZE}...")
all_embs = np.zeros((n_docs, dim), dtype=np.float32)

with torch.no_grad():
    for i, enc in enumerate(loader):
        enc  = {k: v.to(DEVICE) for k, v in enc.items()}
        out  = model(**enc)
        embs = average_pool(out.last_hidden_state, enc["attention_mask"])
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        start = i * BATCH_SIZE
        end   = min(start + BATCH_SIZE, n_docs)
        all_embs[start:end] = embs.cpu().float().numpy()
        if i % 20 == 0:
            print(f"  {end:,}/{n_docs:,}")

np.save(os.path.join(index_dir, "embeddings.npy"), all_embs)
print("Embeddings saved.")

# ── build FAISS IVF index ──────────────────────────────────────────────────────
print(f"Building IVFFlat index (nlist={NLIST})...")
quantizer = faiss.IndexFlatIP(dim)
index     = faiss.IndexIVFFlat(quantizer, dim, NLIST, faiss.METRIC_INNER_PRODUCT)

# use GPU for training if available
if DEVICE == "cuda":
    res       = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
    index_gpu.train(all_embs)
    index_gpu.add(all_embs)
    index     = faiss.index_gpu_to_cpu(index_gpu)
else:
    index.train(all_embs)
    index.add(all_embs)

faiss.write_index(index, os.path.join(index_dir, "ivf.index"))
print(f"Index written to {index_dir}ivf.index  ({index.ntotal:,} vectors)")
