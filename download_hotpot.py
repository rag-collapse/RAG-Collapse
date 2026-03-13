from datasets import load_dataset

cache_dir = "/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache/"

corpus = load_dataset("mteb/hotpotqa", "corpus", cache_dir=cache_dir)
queries = load_dataset("mteb/hotpotqa", "queries", cache_dir=cache_dir)
qrels = load_dataset("mteb/hotpotqa", "default", cache_dir=cache_dir)

train_qids = set(qrels["train"]["query-id"])
test_qids = set(qrels["test"]["query-id"])
dev_qids = set(qrels["dev"]["query-id"])

all_queries = queries["queries"]
train_queries = all_queries.filter(lambda x: x["_id"] in train_qids)
test_queries  = all_queries.filter(lambda x: x["_id"] in test_qids)
dev_queries   = all_queries.filter(lambda x: x["_id"] in dev_qids)

train_queries.save_to_disk(f"{cache_dir}/queries/train")
test_queries.save_to_disk(f"{cache_dir}/queries/test")
dev_queries.save_to_disk(f"{cache_dir}/queries/val")

# train_queries = load_from_disk(f"{cache_dir}/queries/train")
# test_queries  = load_from_disk(f"{cache_dir}/queries/test")
# val_queries   = load_from_disk(f"{cache_dir}/queries/val")