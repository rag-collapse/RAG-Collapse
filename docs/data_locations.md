# Raw Experiment Data Locations (Unity HPC)

This document maps **exactly** where every raw experiment JSON file is stored on
Unity, who owns it, and which script produced it. Verified by walking the
filesystem on `ssh unity` (not just from Slack), so paths reflect what is
actually on disk.

> All paths live under two roots:
> - **Shared project storage:** `/work/pi_dagarwal_umass_edu/project_4/file_storage/<user>/`
> - **Ozel's scratch space:** `/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/<user>/`
>
> The two are mostly mirrors; `/work` is the durable copy and what you should
> cite. Scratch holds extra reranker/oracle/finetuned runs not yet copied to `/work`.

---

## 1. Directory layout & naming conventions

Each user directory contains three parallel output trees, one per pipeline stage:

| Tree | Produced by | Contents |
|------|-------------|----------|
| `experiment_outputs/` | `pipeline.py` (`scripts/pipeline.sh`) | Raw per-round generations + tool calls. The source of truth. |
| `evaluation_outputs/` | `evaluation.py` (`scripts/eval.sh`) | Text-similarity metrics per round: ROUGE-1/2/L, TES, same-answer %, AI-reference %, unique words. |
| `entity_extraction_output/` | `entity_extraction.py` (`scripts/entity_extraction.sh`) | Canonical entity sets per round → **collapse rate** is computed from these. |

Within each tree the path is `…/<org>/<model>/<file>` and the **filename prefix
encodes the experiment variant**:

| Filename pattern | Variant | Dataset |
|------------------|---------|---------|
| `local_replace_all_*`  | Replace All (hybrid) | Graphite |
| `local_replace_one_*`  | Replace One | Graphite |
| `local_search_*`       | Search | Graphite |
| `local_agentic_rag_*`  | Agentic RAG | Graphite |
| `rerank_lambda0.7_*`   | Rerank (λ=0.7 is the reported one; 0.1/0.5 are sweeps) | Graphite |
| `cpu_<variant>_paraphrased_*` | Search/Replace-One/Hybrid **Paraphrase** | Graphite |
| `hotpot_<variant>_*`   | any variant | HotpotQA |
| `*_oracle_*`, `*_desklib_*`, `*_finetuned_*` | reranker ablations | HotpotQA / Graphite |

File suffix: `_eval.json` (evaluation), `_entity_results.json` (entity),
`.json` (raw pipeline), `.checkpoint.json` (resumable mid-run state, ignore).

---

## 2. Who produced what (ownership)

| Owner (`<user>`) | What they ran |
|------------------|---------------|
| `ffatima_umass_edu`     | Qwen-14B & Mistral-7B **baselines** (`…/para_off/`), **Paraphrase** runs for all models, and the shared `reranker/` Qwen-14B rerank outputs |
| `ratirastogi_umass_edu` | **DeepSeek-R1-7B** & **Llama-3.1-8B** baselines; Qwen-7B / 1.5B / Qwen3-4B baselines |
| `rsenapati_umass_edu`   | **Agentic RAG** for all four main models |
| `oyilmazel_umass_edu`   | **HotpotQA** runs (all models), reranker **oracle / desklib / finetuned** ablations, `gpt-oss-20b` |
| `reranker` (shared dir) | Qwen-14B **Rerank** λ-sweep (0.1 / 0.5 / 0.7), full-access permissions |

---

## 3. Graphite dataset: the data behind the aggregate collapse plots

These are the files used by `agentic-rag-visualization.ipynb` for the combined
multi-model collapse bar chart. For collapse you need the **`entity_extraction_output`**
files; eval files give the per-round similarity metrics.

### 3.1 Qwen2.5-14B-Instruct  (primary model, has all 6 variants)

**Replace All**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/evaluation_outputs/Qwen/Qwen2.5-14B-Instruct/para_off/local_replace_all_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/entity_extraction_output/Qwen/Qwen2.5-14B-Instruct/local_replace_all_entity_results.json`

**Replace One**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/evaluation_outputs/Qwen/Qwen2.5-14B-Instruct/para_off/local_replace_one_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/entity_extraction_output/Qwen/Qwen2.5-14B-Instruct/local_replace_one_entity_results.json`

**Search**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/evaluation_outputs/Qwen/Qwen2.5-14B-Instruct/para_off/local_search_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/entity_extraction_output/Qwen/Qwen2.5-14B-Instruct/local_search_entity_results.json`

**Rerank (λ0.7)**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/reranker/evaluation_outputs/Qwen__Qwen2.5-14B-Instruct/rerank_lambda0.7_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/reranker/entity_extraction_output/Qwen__Qwen2.5-14B-Instruct/rerank_lambda0.7_entity_results.json`

**Search Paraphrase**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/evaluation_outputs/Qwen/Qwen2.5-14B-Instruct/rerun-paraphrase-cpu/cpu_search_paraphrased_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/entity_extraction_output/Qwen/Qwen2.5-14B-Instruct/rerun-paraphrase/cpu_search_paraphrased_entity_results.json`

**Agentic RAG**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/evaluation_outputs/Qwen/Qwen2.5-14B-Instruct/local_agentic_rag_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/entity_extraction_output/Qwen/Qwen2.5-14B-Instruct/local_agentic_rag_entity_results.json`

> **Note on Search Paraphrase entity files:** the entity results live in the
> `rerun-paraphrase/` subfolder (filename `cpu_search_paraphrased_entity_results.json`),
> **not** `rerun-paraphrase-cpu-server/` which Slack referenced. That folder is
> empty on disk. The eval files are under `rerun-paraphrase-cpu/`. Other paraphrase
> variants (`cpu_replace_one_*`, `cpu_hybrid_*`) exist alongside them.

> Reranker λ-sweep companions (if you want 0.1 / 0.5): same dirs, swap `0.7`→`0.1`/`0.5`.
> - `/work/pi_dagarwal_umass_edu/project_4/file_storage/reranker/evaluation_outputs/Qwen__Qwen2.5-14B-Instruct/rerank_lambda0.1_eval.json`
> - `/work/pi_dagarwal_umass_edu/project_4/file_storage/reranker/evaluation_outputs/Qwen__Qwen2.5-14B-Instruct/rerank_lambda0.5_eval.json`

### 3.2 DeepSeek-R1-Distill-Qwen-7B  (baselines + Agentic RAG)

**Replace All**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ratirastogi_umass_edu/evaluation_outputs/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/local_replace_all_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ratirastogi_umass_edu/entity_extraction_output/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/local_replace_all_entity_results.json`

**Replace One**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ratirastogi_umass_edu/evaluation_outputs/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/local_replace_one_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ratirastogi_umass_edu/entity_extraction_output/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/local_replace_one_entity_results.json`

**Search**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ratirastogi_umass_edu/evaluation_outputs/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/local_search_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ratirastogi_umass_edu/entity_extraction_output/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/local_search_entity_results.json`

**Agentic RAG**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/evaluation_outputs/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/local_agentic_rag_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/entity_extraction_output/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/local_agentic_rag_entity_results.json`

**Paraphrase (extra, Search variant)**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/evaluation_outputs/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/rerun-paraphrase/cpu_search_paraphrased_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/entity_extraction_output/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/rerun-paraphrase/cpu_search_paraphrased_entity_results.json`

### 3.3 Llama-3.1-8B-Instruct  (baselines + Agentic RAG)

**Replace All**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ratirastogi_umass_edu/evaluation_outputs/meta-llama/Llama-3.1-8B-Instruct/local_replace_all_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ratirastogi_umass_edu/entity_extraction_output/meta-llama/Llama-3.1-8B-Instruct/local_replace_all_entity_results.json`

**Replace One**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ratirastogi_umass_edu/evaluation_outputs/meta-llama/Llama-3.1-8B-Instruct/local_replace_one_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ratirastogi_umass_edu/entity_extraction_output/meta-llama/Llama-3.1-8B-Instruct/local_replace_one_entity_results.json`

**Search**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ratirastogi_umass_edu/evaluation_outputs/meta-llama/Llama-3.1-8B-Instruct/local_search_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ratirastogi_umass_edu/entity_extraction_output/meta-llama/Llama-3.1-8B-Instruct/local_search_entity_results.json`

**Agentic RAG**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/evaluation_outputs/meta-llama/Llama-3.1-8B-Instruct/local_agentic_rag_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/entity_extraction_output/meta-llama/Llama-3.1-8B-Instruct/local_agentic_rag_entity_results.json`

**Paraphrase (extra, Search variant)**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/evaluation_outputs/meta-llama/Llama-3.1-8B-Instruct/rerun-paraphrase/cpu_search_paraphrased_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/entity_extraction_output/meta-llama/Llama-3.1-8B-Instruct/rerun-paraphrase/cpu_search_paraphrased_entity_results.json`

### 3.4 Mistral-7B-Instruct-v0.3  (baselines + Agentic RAG)

**Replace All**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/evaluation_outputs/mistralai/Mistral-7B-Instruct-v0.3/para_off/local_replace_all_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/entity_extraction_output/mistralai/Mistral-7B-Instruct-v0.3/local_replace_all_entity_results.json`

**Replace One**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/evaluation_outputs/mistralai/Mistral-7B-Instruct-v0.3/para_off/local_replace_one_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/entity_extraction_output/mistralai/Mistral-7B-Instruct-v0.3/local_replace_one_entity_results.json`

**Search**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/evaluation_outputs/mistralai/Mistral-7B-Instruct-v0.3/para_off/local_search_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/entity_extraction_output/mistralai/Mistral-7B-Instruct-v0.3/local_search_entity_results.json`

**Agentic RAG**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/evaluation_outputs/mistralai/Mistral-7B-Instruct-v0.3/local_agentic_rag_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/entity_extraction_output/mistralai/Mistral-7B-Instruct-v0.3/local_agentic_rag_entity_results.json`

**Paraphrase (extra, Search variant)**
- eval:   `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/evaluation_outputs/mistralai/Mistral-7B-Instruct-v0.3/rerun-paraphrase/cpu_search_paraphrased_eval.json`
- entity: `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/entity_extraction_output/mistralai/Mistral-7B-Instruct-v0.3/rerun-paraphrase/cpu_search_paraphrased_entity_results.json`

> **Mistral Rerank** on the Graphite dataset is **not available** (Slack: "We
> don't have reranker mistral with the HotpotQA finetuning for Graphite Dataset").
> A Mistral rerank eval exists only in scratch
> (`/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/ffatima_umass_edu/evaluation_outputs/mistralai__Mistral-7B-Instruct-v0.3/rerank_lambda0.7_eval.json`)
> but has no matching entity file, so it can't contribute a collapse bar.

---

## 4. Raw pipeline outputs (for Agentic RAG tool-call analysis)

The notebook's tool-calls-per-round plot reads the raw pipeline file (not eval/entity):

```
/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/experiment_outputs/Qwen/Qwen2.5-14B-Instruct/local_agentic_rag.json
/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/experiment_outputs/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/local_agentic_rag.json
/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/experiment_outputs/meta-llama/Llama-3.1-8B-Instruct/local_agentic_rag.json
/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/experiment_outputs/mistralai/Mistral-7B-Instruct-v0.3/local_agentic_rag.json
```

---

## 5. HotpotQA dataset (separate from Graphite)

All under `oyilmazel_umass_edu`, prefix `hotpot_`. Models: Qwen-14B, Qwen-7B,
Qwen3-4B, DeepSeek-R1-7B, Llama-3.1-8B, Mistral-7B.

Template:
```
/work/pi_dagarwal_umass_edu/project_4/file_storage/oyilmazel_umass_edu/evaluation_outputs/hotpotqa/<org>/<model>/hotpot_<variant>_hotpot_eval.json
/work/pi_dagarwal_umass_edu/project_4/file_storage/oyilmazel_umass_edu/experiment_outputs/hotpotqa/<org>/<model>/hotpot_<variant>.json
```

Example (Qwen2.5-14B, search):
```
/work/pi_dagarwal_umass_edu/project_4/file_storage/oyilmazel_umass_edu/evaluation_outputs/hotpotqa/Qwen/Qwen2.5-14B-Instruct/hotpot_search_hotpot_eval.json
/work/pi_dagarwal_umass_edu/project_4/file_storage/oyilmazel_umass_edu/experiment_outputs/hotpotqa/Qwen/Qwen2.5-14B-Instruct/hotpot_search.json
```

`<variant>` ∈ `replace_all`, `replace_one`, `search`, `rerank_lambda{0.1,0.5,0.7,1.0}`.
Files ending `_hotpot_eval_summary.json` are the aggregated EM/F1 summaries.
Reranker **finetuned / oracle / desklib** HotpotQA ablations are in **scratch only**:
```
/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/oyilmazel_umass_edu/evaluation_outputs/hotpotqa/<org>/<model>/hotpot_rerank_lambda0.7_{finetuned,oracle,desklib}_hotpot_eval.json
```

---

## 6. Other / smaller models (not in main plots)

| Model | Owner | Representative path |
|-------|-------|---------------------|
| `Qwen/Qwen2.5-7B-Instruct`  | ratirastogi | `/work/pi_dagarwal_umass_edu/project_4/file_storage/ratirastogi_umass_edu/evaluation_outputs/Qwen/Qwen2.5-7B-Instruct/local_search_eval.json` |
| `Qwen/Qwen2.5-1.5B-Instruct`| ratirastogi | `/work/pi_dagarwal_umass_edu/project_4/file_storage/ratirastogi_umass_edu/evaluation_outputs/Qwen/Qwen2.5-1.5B-Instruct/local_search_eval.json` |
| `Qwen/Qwen3-4B-Instruct(-2507)` | oyilmazel / ratirastogi | `/work/pi_dagarwal_umass_edu/project_4/file_storage/oyilmazel_umass_edu/evaluation_outputs/Qwen/Qwen3-4B-Instruct/local_search_eval.json` |
| `openai/gpt-oss-20b` | oyilmazel | `/work/pi_dagarwal_umass_edu/project_4/file_storage/oyilmazel_umass_edu/evaluation_outputs/openai/gpt-oss-20b/local_search_eval.json` |
| `50-questions-run/*` | ffatima / rsenapati | `/work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/entity_extraction_output/Qwen/50-questions-run/Qwen2.5-14B-Instruct/`, early 50-question smoke runs, **not** the full benchmark |

---

## 7. Producing scripts (repo → output mapping)

From PR history on this repo (`main` + `gepa` merge, PR #28):

| Output tree | Script (sbatch) | Python entrypoint |
|-------------|-----------------|-------------------|
| `experiment_outputs/` | `scripts/pipeline.sh` | `pipeline.py` → `pipeline/` package (`feedback_loop.py`, `model_runner.py`, `retrieval.py`, …) |
| `evaluation_outputs/` | `scripts/eval.sh` | `evaluation.py` |
| `entity_extraction_output/` | `scripts/entity_extraction.sh` | `entity_extraction.py` |
| GEPA prompt optimization | `scripts/gepa_optimization.sh`, `scripts/gepa_prepare_dataset.sh` | (GEPA driver) |
| GEPA pipeline (new, non-agentic, `--use-gepa-prompt`) | `scripts/gepa_pipeline/run_*.sh` + `server_*.sh` | `pipeline.py --use-gepa-prompt` |
| vLLM model servers | `scripts/start_llm_server_*.sh` | n/a |

Variant selection is a CLI flag on `pipeline.py` (`--mode replace_all|replace_one|search|agentic_rag`);
`--use-gepa-prompt` swaps in `GEPA_RAG_GENERATION_SYSTEM_PROMPT` from `formatters.py`.

---

## 8. Quick-copy: gather everything into your own scratch

To pull the canonical Graphite collapse data into
`/scratch4/workspace/rsenapati_umass_edu-rag-collapse` (you have read access to
all `/work` paths above):

```bash
WORK=/work/pi_dagarwal_umass_edu/project_4/file_storage
DST=/scratch4/workspace/rsenapati_umass_edu-rag-collapse/collected
mkdir -p "$DST"
# example: copy preserving relative tree
rsync -a --relative \
  "$WORK/./ffatima_umass_edu/entity_extraction_output/Qwen/Qwen2.5-14B-Instruct/" \
  "$WORK/./ratirastogi_umass_edu/entity_extraction_output/" \
  "$WORK/./rsenapati_umass_edu/entity_extraction_output/" \
  "$WORK/./reranker/" \
  "$DST/"
```

(Adjust the source list to whatever subset you need; `--relative` keeps the
`<user>/.../<model>/` structure so nothing collides.)
