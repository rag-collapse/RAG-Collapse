# RAG-Collapsement-on-Self-Refined-Generation

Studies **RAG collapse**: a model answers from retrieved context, its answers are turned back
into "documents" and fed in as context for the next round, and over many rounds the answers
degenerate (lose entity/lexical diversity, converge to the same text). This repo runs that
self-refinement loop under several document-update regimes, measures the collapse, and tests
mitigations (reranking, paraphrasing, agentic retrieval, and a GEPA-optimized system prompt).

End-to-end workflow:

```
pipeline.py            evaluation.py              entity_extraction.py
(run the loop)   →     (text-similarity metrics)  (entity collapse metrics)
experiment_outputs/    evaluation_outputs/        entity_extraction_output/
                              │
                              ▼
                 visualization.ipynb / agentic-rag-visualization.ipynb
```

## Documentation

| Doc | What it covers |
|---|---|
| [`docs/unity_setup.md`](docs/unity_setup.md) | Unity HPC scratch-space setup + job submission workflow |
| [`docs/gepa_prompt_optimization_plan.md`](docs/gepa_prompt_optimization_plan.md) | GEPA system-prompt optimization: dataset, scoring, adapter, how to run |
| [`docs/gepa_flowchart.md`](docs/gepa_flowchart.md) | GEPA architecture & metrics as mermaid diagrams |
| [`docs/data_locations.md`](docs/data_locations.md) | Where every raw experiment JSON lives on Unity |
| [`scripts/all_experiments_README.md`](scripts/all_experiments_README.md) | The consolidated `all_experiments/` data tree (layout + scope) |
| [`docs/entity_extraction_comparison.md`](docs/entity_extraction_comparison.md) | Entity-extraction code vs the `collapse-randomness-research` repo |
| [`docs/pipeline_comparison.md`](docs/pipeline_comparison.md) | Full pipeline (baselines + metrics) vs `collapse-randomness-research` |
| [`docs/misinfo_error_compounding.md`](docs/misinfo_error_compounding.md) | Misinformation error-compounding experiment (HotpotQA) + `scripts/hotpot-misinfo/` |
| [`docs/hotpotqa_diverse_synth.html`](docs/hotpotqa_diverse_synth.html) · [interactive artifact](https://claude.ai/code/artifact/0b92b522-4718-4496-bb94-8d7d2079170a) | Round-0 distractor experiments — `diverse_synth` + `equal_diverse_synth` + `shuffled_diverse_synth` — with 3-variant collapse results, metric definitions & figures. Plots via [`distractor_sweep_visualization.ipynb`](distractor_sweep_visualization.ipynb). |
| [`docs/cross_model_baseline.html`](docs/cross_model_baseline.html) · [interactive artifact](https://claude.ai/code/artifact/2dd7115f-3457-44e8-8914-a088246f773e) | `cross-model-baseline` — a main model reads documents synthesized from a *different* (side) model's answers, plus the GPU-scheduling hack (`bf16` constraint + `DOC_ON_MAIN` 2-GPU mode) that got it to run on Unity. |
| [`visualization_outputs/README.md`](visualization_outputs/README.md) | Plots and how to regenerate them |

### Interactive HTML docs (published Claude artifacts)

| Doc | Artifact |
|---|---|
| `docs/cross_model_baseline.html` | https://claude.ai/code/artifact/2dd7115f-3457-44e8-8914-a088246f773e |
| `docs/hotpotqa_diverse_synth.html` (diverse / equal / shuffled) | https://claude.ai/code/artifact/0b92b522-4718-4496-bb94-8d7d2079170a |
| `docs/distractor_experiment.html` | https://claude.ai/code/artifact/38bd9c07-e6ea-42eb-833b-57b1adea3d56 |
| `docs/hotpotqa_experiments.html` | https://claude.ai/code/artifact/d9929aaf-87b3-432d-b3c6-86a1fd3250a5 |
| `docs/hotpotqa_smoke_results.html` | https://claude.ai/code/artifact/4dca4513-db7e-46db-b9c9-cec021f7e949 |

## Setup

```bash
module load conda/latest          # on HPC
conda create -n ragenv python=3.11
conda activate ragenv
pip install -r requirements.txt    # vllm, transformers, sentence-transformers, litellm, gepa, rouge-score, nltk, faiss-cpu, ...
```

On Unity, point the HuggingFace cache at scratch so compute nodes don't re-download models
(`export HF_HOME=/scratch.../hf_cache`); see [`docs/unity_setup.md`](docs/unity_setup.md).

## The collapse experiment

The loop is set by `--pipeline-variant`; each variant differs only in how the document corpus
is updated between rounds:

- **Replace All** (`hybrid` + `--num-synth-docs 10 --num-db-docs 0`): round 0 uses the question's
  references; from round 1 the entire context is 10 synthetic docs built from the model's own prior
  answers. **10 rounds.**
- **Replace One** (`replace_one`): start with the references (≤10); each round replace exactly one
  slot with a new AI-generated doc. **20 rounds.** (`--stop-if-converged` ends early after 4 stable rounds.)
- **Search** (`search`): a vector store is seeded with the references; each round retrieves the
  top-k chunks, and each new answer is added to the store as a competing document. **30 rounds.**
- **Agentic RAG** (`agentic_rag`): same growing store as `search`, but the model is given a
  `retrieve(query)` tool and decides what to fetch itself (up to `--agentic-max-tool-calls`). **30 rounds.**

**Feedback step:** each answer is expanded into a web-style article via the "create document"
prompt in `formatters.py`; that text becomes the synthetic document(s) for the next round.

**System prompt:** by default the baseline `_RAG_GENERATION_SYSTEM_PROMPT` is used. Pass
`--use-gepa-prompt` to instead use `GEPA_RAG_GENERATION_SYSTEM_PROMPT` (the GEPA-optimized prompt) —
see [GEPA prompt optimization](#gepa-prompt-optimization).

**Output:** one JSON per run with `experiment_metadata`, `questions`, and per-iteration
`documents` + `runs` (e.g. `local_replace_all.json`, `local_search.json`, `local_agentic_rag.json`).

## Running the pipeline

Three model modes (`--model-mode`):
- **`server`** (recommended): HTTP client to a running vLLM OpenAI-compatible server. Pass
  `--vllm-api-base <url>` or export `VLLM_API_BASE`. No GPU in the pipeline process.
- **`local`**: loads the model in-process via vLLM. Needs a GPU in the job.
- **`api`**: a proprietary API (e.g. GPT-4) via LiteLLM. Set `API_KEY`.

Only `pipeline.py` (server mode) and `llm_service/inference_example.py` need `VLLM_API_BASE`;
`evaluation.py` and `entity_extraction.py` load their own (small) models locally.

### Step 1 — start a vLLM server (server mode)

```bash
vllm serve <model> --served-model-name qwen2.5-14b --tensor-parallel-size 2 --port 5150 --host 0.0.0.0
export VLLM_API_BASE="http://<fqdn>:5150/v1"     # or http://localhost:5150/v1
```

For the **`agentic_rag`** variant the server must enable tool calling with the parser for your
model family (verified against current vLLM):

| Model family | flags |
|---|---|
| Qwen2.5 | `--enable-auto-tool-choice --tool-call-parser hermes` |
| Llama-3.1 | `--enable-auto-tool-choice --tool-call-parser llama3_json` |
| Mistral | `--tokenizer-mode mistral --enable-auto-tool-choice --tool-call-parser mistral` |
| DeepSeek-R1 | add `--reasoning-parser deepseek_r1` (strips `<think>` tokens) |

Ready-made server scripts: `scripts/start_llm_server_*.sh`.

### Step 2 — run

```bash
mkdir -p experiment_outputs
python -u pipeline.py \
  --model-mode server --vllm-api-base "$VLLM_API_BASE" --model-name qwen2.5-14b \
  --dataset-path datasets/umass_data.entity.chatgpt.50.jsonl \
  --output-path experiment_outputs/output.json \
  --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 \
  --num-runs 10 --chars-per-doc 400 --max-questions 8
```

**Key parameters:**
- `--pipeline-variant`: `hybrid` (Replace All with `--num-synth-docs 10 --num-db-docs 0`), `replace_one`, `search`, `agentic_rag`
- `--dataset-path`: input JSONL (`datasets/umass_data.entity.chatgpt.{50,400}.jsonl`)
- `--num-runs`: responses per round (default 10); `--max-iterations`: cap rounds for quick tests; `--max-questions`: limit questions
- `--chars-per-doc`: per-document character cap
- `--use-gepa-prompt`: use the GEPA-optimized `GEPA_RAG_GENERATION_SYSTEM_PROMPT` instead of the baseline
- `--enable-citations`: enable citation generation (off by default)
- `--paraphrase-reference-docs`: rewrite human reference docs through the create-document prompt before round 0 (normalizes human vs AI style)
- `--doc-model-mode` / `--doc-vllm-api-base` / `--doc-model-name` / ...: route the expand-to-article step to a separate model
- search-only: `--search-embedding-mode {local,api}`, `--search-embedding-model` (default `all-MiniLM-L6-v2`), `--search-top-k` (10), `--search-chunk-size` (500), `--search-chunk-overlap` (50)
- agentic-only: `--agentic-max-tool-calls` (10)

**Quick smoke test** (1 question, 2 rounds) — swap `--pipeline-variant` per variant:
```bash
python -u pipeline.py --model-mode server --vllm-api-base "$VLLM_API_BASE" --model-name qwen2.5-14b \
  --pipeline-variant search --dataset-path datasets/umass_data.entity.chatgpt.50.jsonl \
  --output-path experiment_outputs/test_search.json --max-questions 1 --max-iterations 2
```

## Evaluation & entity extraction

After the pipeline, two scripts compute the collapse metrics from the experiment JSON.

**`evaluation.py`** — text-similarity metrics per round: pairwise embedding cosine similarity,
sentence-level TES, ROUGE-1/2/L, `unique_words`, `ai_reference_percentage` (fraction of context
docs that are AI-generated), `ai_citation_percentage`, and `same_answer_percentage` (an LLM
paraphrase judge, model from `SAME_ANSWER_MODEL_NAME`, sampled over 10 answer pairs).
```bash
python evaluation.py <experiment.json> <eval.json> [--cache-dir <hf_cache>]
# or: sbatch scripts/eval.sh   # set MODEL_SUBDIR and the `for base in ...` variant loop inside
```

**`entity_extraction.py`** — entity-collapse metrics: extracts answer entities (LLM), clusters
to canonical forms, recovers missed mentions, and reports `unique_entities` per round and
pairwise entity-set similarity (mean/min/max/std).
```bash
python entity_extraction.py --model-mode local --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --experiment-files experiment_outputs/.../local_search.json --output-dir entity_extraction_output/...
# or: sbatch scripts/entity_extraction.sh   # set MAIN_MODEL / ENTITY_MODEL / variant inside
```

## GEPA prompt optimization

`gepa_optimization/` evolves the RAG system prompt to resist collapse while staying faithful,
using GEPA's reflective Pareto search over **anti-collapse** × **quality**. Full details in
[`docs/gepa_prompt_optimization_plan.md`](docs/gepa_prompt_optimization_plan.md) and
[`docs/gepa_flowchart.md`](docs/gepa_flowchart.md).

```bash
# 1. Build the dataset (UMass + HotpotQA via E5+FAISS retrieval) — GPU, one-time
sbatch scripts/gepa_prepare_dataset.sh

# 2. Run optimization (CPU; all generation via the keymaker API) — env-var driven
export API_KEY="your-keymaker-key"
sbatch --export=ALL scripts/gepa_optimization.sh

# 3. Rank candidates, then write the best prompt into formatters.py
python gepa_optimization/parse_results.py --run-dir gepa_runs/rag_system_prompt_<id>
python gepa_optimization/apply_best_prompt.py --prompt '<optimized prompt>'
```

Then run the pipeline with `--use-gepa-prompt` and re-run evaluation/entity-extraction to compare
against a baseline run. The **`scripts/gepa_pipeline/`** folder has ready run + server scripts to
sweep the four models (Qwen2.5-14B, DeepSeek-R1-7B, Llama-3.1-8B, Mistral-7B) with the GEPA prompt;
its server scripts are non-agentic (no tool-calling flags).

## Consolidating results

`scripts/consolidate_experiments.{py,sh}` gathers every experiment/eval/entity JSON across users
into one tree, `all_experiments/<dataset>/<method>[/<config>]/<output_tree>/...`. See
[`scripts/all_experiments_README.md`](scripts/all_experiments_README.md) for the layout/scope and
[`docs/data_locations.md`](docs/data_locations.md) for the source locations.

```bash
sbatch scripts/consolidate_experiments.sh
```

## Visualization

`visualization.ipynb` (baseline variants) and `agentic-rag-visualization.ipynb` (adds Agentic RAG
+ a multi-model collapse-rate bar chart) render per-round plots from the eval/entity JSONs into
`visualization_outputs/`. See [`visualization_outputs/README.md`](visualization_outputs/README.md).

## Running on SLURM

All batch scripts are in [`scripts/`](scripts/) (run `mkdir -p logs` first):

| Script | Purpose |
|---|---|
| `pipeline.sh` | run the experiment loop (server mode by default; edit `MODEL`, `DATASET`, `EXTRA`) |
| `eval.sh` | run `evaluation.py` over selected variants (edit `MODEL_SUBDIR` + the `for base` loop) |
| `entity_extraction.sh` | run `entity_extraction.py` (edit `MAIN_MODEL` / `ENTITY_MODEL`) |
| `inference.sh` | smoke-test the LLM service (`llm_service/inference_example.py`) |
| `start_llm_server_*.sh` | launch a vLLM server per model family |
| `gepa_prepare_dataset.sh`, `gepa_optimization.sh` | GEPA dataset prep (GPU) + optimization (CPU) |
| `gepa_pipeline/{run,server}_*.sh` | run the 4-model sweep with the GEPA prompt |
| `consolidate_experiments.sh` | build the consolidated `all_experiments/` tree |

```bash
# pipeline (start a vLLM server and export VLLM_API_BASE first)
sbatch --export=ALL scripts/pipeline.sh        # writes experiment_outputs/$MODEL/local_*.json

# monitor
squeue --me
tail -f logs/pipeline_<JOB_ID>.out
scancel <JOB_ID>
```

`scripts/pipeline.sh` defaults: `MODEL=Qwen/Qwen2.5-14B-Instruct`,
`DATASET=datasets/umass_data.entity.chatgpt.400.jsonl`, gpu partition, 48 h, 1 GPU
(`vram40|vram48|vram80`), 48 GB, 4 CPUs. Switch model by editing `MODEL`; outputs land under a
matching `$MODEL` subdirectory.

## Example pipeline

![Pipeline](example.png)
