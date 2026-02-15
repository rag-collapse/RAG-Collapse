# Pipeline progress

### 15th Feb — Pipeline variants
- **README** updated: pipeline details, parameters, testing commands, SLURM section; three variants and new output paths; job name `pipeline`; `HF_HOME` / `HF_HUB_CACHE` for cache.

### 14th Feb — Pipeline variants
- **Three document-setting variants** only: Replace All (hybrid 10/0), Replace One (one slot replaced per round), Search (vector retrieval). Script runs them one by one; outputs: `local_replace_all.json`, `local_replace_one.json`, `local_search.json`.
- **Config** (`config.py`): `PIPELINE_VARIANTS` = hybrid, replace_one, search. Rounds: 10 / 20 / 30. Replace One truncates refs to 10; `ROUNDS_REPLACE_ALL`, `ROUNDS_REPLACE_ONE`, `ROUNDS_SEARCH`. Docstrings use “document setting” and describe hybrid vs replace_one clearly.
- **Single context path**: `context_builder.py` has `get_next_documents()` and `get_initial_documents_hybrid()` / `get_initial_documents_replace_one()`; `pipeline.py` uses these for all variants.
- **Hybrid** subsumes configurable “replace” behavior via `--num-synth-docs` and `--num-db-docs`. **Search**: local embeddings by default (SentenceTransformer); 30 rounds; `ChunkedRetrievalStore`. **Data loader**: `prepare_dataset()` filters by min citations and truncates refs only for replace_one (max 10).
- **Pipeline script** (`scripts/pipeline.sh`): cache dir (`HF_HOME` / `HF_HUB_CACHE`); three `run_local` calls for the three variants.

### 8th Feb — Replace One (document setting)
- **Replace One in code**: `next_docs_replace_one()` in `feedback_loop.py` (slot = `iteration % len(current_docs)`); `get_initial_documents_replace_one()` and replace_one branch in `context_builder.py`; pipeline sets initial docs and calls `get_next_documents()` for replace_one. Data loader truncates refs to 10 for replace_one only.
- **Optional convergence**: `--stop-if-converged`; if the same answer-length signature repeats for 4 consecutive rounds, iteration stops early for that question (no new evaluation metrics).

---

### Start — Replace All pipeline
- Pipeline that runs the **Replace All** setup: context each round = all synthetic (from model generations). Hybrid variant with `--num-synth-docs 10 --num-db-docs 0`; 10 rounds. Model answers from retrieved context; answers are turned into documents and fed back for the next round.
