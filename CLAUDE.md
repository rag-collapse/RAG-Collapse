# CLAUDE.md

Guidance for working in this repo. See `README.md` for the user-facing walkthrough and `docs/` for deep dives.

## What this is

Research code studying **RAG collapse**: a model answers from retrieved context, its answers are
turned back into "documents" and fed in as context next round, and over many rounds the answers
degenerate (lose entity/lexical diversity, converge to the same text). The repo runs that
self-refinement loop under several document-update regimes, measures collapse, and tests
mitigations (reranking, paraphrasing, agentic retrieval, and a GEPA-optimized system prompt).

Pipeline of work: `pipeline.py` (run loop) → `evaluation.py` (text metrics) → `entity_extraction.py`
(entity metrics) → `visualization*.ipynb`. Outputs go to `experiment_outputs/`, `evaluation_outputs/`,
`entity_extraction_output/` (per-model subdirs).

## Layout

| Path | Role |
|---|---|
| `pipeline.py` + `pipeline/` | the collapse loop (data_loader, retrieval, model_runner, context_builder, feedback_loop, prompt_builder, ai_detector, config, output_writer) |
| `evaluation.py` | text-similarity metrics (ROUGE/TES/cosine, unique_words, ai_reference%, same_answer judge) |
| `entity_extraction.py` | entity-collapse metrics (unique entities, entity similarity) |
| `formatters.py` | all prompts: `_RAG_GENERATION_SYSTEM_PROMPT` (baseline), `GEPA_RAG_GENERATION_SYSTEM_PROMPT` (GEPA seed/optimized), create-document, agentic tool spec |
| `llm_service/` | LLM backends: `OpenSourceLLM` (vLLM), `ProprietaryLLM` (LiteLLM/keymaker), `ServerLLM` (HTTP vLLM), `CommonLLM` interface, `EmbeddingModel` |
| `gepa_optimization/` | GEPA prompt optimization package (prepare_dataset, rag_adapter, scoring, pipeline_simulator, run_optimization, parse_results, apply_best_prompt) |
| `hotpot_pipeline.py`, `hotpot_evaluation.py`, `build_hotpotqa_index.py`, `download_hotpot.py` | HotpotQA-dataset variants of the loop/eval |
| `rerank_mitigation_pipeline.py`, `rerank_mitigation_hotpot_pipeline.py` | reranker-mitigation pipelines |
| `scripts/` | SLURM batch scripts (see below) |
| `datasets/` | input JSONL (`umass_data.entity.chatgpt.{50,400}.jsonl`) |
| `docs/` | see Documentation index |

## Setup & environment

- Conda env **`ragenv`** (python 3.11), `pip install -r requirements.txt` (vllm, transformers,
  sentence-transformers, litellm, gepa, rouge-score, nltk, faiss-cpu, …).
- On Unity HPC, point HF cache at scratch: `export HF_HOME=/scratch.../hf_cache`. See `docs/unity_setup.md`.
- Keymaker API key via `export API_KEY=...` (LiteLLM proxy `https://thekeymaker.umass.edu/`).
  **Never commit API keys.**

## Core commands

```bash
# pipeline (server mode → needs a running vLLM server + VLLM_API_BASE)
python pipeline.py --model-mode server --vllm-api-base "$VLLM_API_BASE" --model-name qwen2.5-14b \
  --pipeline-variant search --dataset-path datasets/umass_data.entity.chatgpt.50.jsonl \
  --output-path experiment_outputs/out.json --num-runs 10 --chars-per-doc 400
#   add --use-gepa-prompt to use GEPA_RAG_GENERATION_SYSTEM_PROMPT instead of the baseline

python evaluation.py <experiment.json> <eval.json> [--cache-dir <hf_cache>]
python entity_extraction.py --model-mode local --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --experiment-files <experiment.json> --output-dir entity_extraction_output/<model>

# GEPA: prepare data (GPU) → optimize (CPU) → rank → apply
sbatch scripts/gepa_prepare_dataset.sh
export API_KEY=...; sbatch --export=ALL scripts/gepa_optimization.sh
python gepa_optimization/parse_results.py --run-dir gepa_runs/rag_system_prompt_<id>
python gepa_optimization/apply_best_prompt.py --prompt '<text>'   # writes GEPA_RAG_GENERATION_SYSTEM_PROMPT

# consolidate all results into all_experiments/ on Unity
sbatch scripts/consolidate_experiments.sh
```

SLURM scripts (run `mkdir -p logs` first): `pipeline.sh`, `eval.sh`, `entity_extraction.sh`,
`inference.sh`, `start_llm_server_*.sh`, `gepa_prepare_dataset.sh`, `gepa_optimization.sh`,
`gepa_pipeline/{run,server}_*.sh` (4-model sweep with the GEPA prompt), `consolidate_experiments.sh`.

## Pipeline variants (`--pipeline-variant`)

| Variant | Rule | Rounds |
|---|---|---|
| `hybrid` (Replace All with `--num-synth-docs 10 --num-db-docs 0`) | entire context replaced by prior answers | 10 |
| `replace_one` | one doc slot replaced per round | 20 |
| `search` | growing vector store; retrieve top-k each round | 30 |
| `agentic_rag` | growing store + model-driven `retrieve()` tool calls | 30 |

Models studied: Qwen2.5-14B, DeepSeek-R1-Distill-Qwen-7B, Llama-3.1-8B, Mistral-7B (default in
`pipeline.sh` is Qwen2.5-14B). Modes: `server` (vLLM HTTP, recommended), `local` (in-process vLLM, GPU), `api` (LiteLLM).

vLLM server tool/reasoning flags for `agentic_rag` (per model family): Qwen → `--tool-call-parser hermes`;
Llama → `llama3_json`; Mistral → `--tokenizer-mode mistral --tool-call-parser mistral`; DeepSeek → add `--reasoning-parser deepseek_r1`. All need `--enable-auto-tool-choice`.

## Conventions

- **AI-doc tagging:** synthetic docs get `doc_id="gen_{iter}_{i}"`, `url="model_generated"`; references get
  `doc_id="ref_{iter}_{i}"`. Every AI-contamination metric keys off these tags (not content analysis).
- **Output schema:** experiment JSON = `{experiment_metadata, questions:[{question_id, question_text,
  iterations:[{iteration_number, documents, runs:[{run_id, answer, citations, ...}]}]}]}`.
- **Datasets:** dataset JSONL records = `{"question", "references":[{"url","text"}]}`.

## Gotchas / caveats (verified against code)

- `evaluation.py` and `entity_extraction.py` do **not** use `VLLM_API_BASE`; they load their own
  (small) models locally and need a GPU. Only `pipeline.py` (server mode) + `inference_example.py` need the server.
- `evaluation.py` `aggregate_statistics.avg_collapse_rate` is **hardcoded to 1** (placeholder) — real
  aggregation across rounds/questions happens in the notebooks, not here.
- `unique_words` (evaluation.py) and `unique_entities` (entity_extraction.py) are the **union across the
  10 runs in a round** (a round-total), NOT a per-run average. The entity `mean_unique_entities_per_round`
  aggregate is a mean **across questions** of those round-totals.
- Entity similarity for an empty-entity answer is scored **0.0** here (treated as diverse); the
  `collapse-randomness-research` repo uses **1.0** (treated as collapsed) — opposite convention. See
  `docs/entity_extraction_comparison.md`.
- GEPA optimizes/writes `GEPA_RAG_GENERATION_SYSTEM_PROMPT` (not `_RAG_GENERATION_SYSTEM_PROMPT`).
- **Known GEPA bugs (unfixed):** `scripts/gepa_optimization.sh` has `export API_KEY=""` right before its
  key check (blanks a `--export=ALL` key); `gepa_optimization/rag_adapter.py` runs the candidate prompt
  through `DOC_GEN_MODEL` because `task_llm` (from `TASK_MODEL`) is built but unused.

## Data

Consolidated results live on Unity at `/work/pi_dagarwal_umass_edu/project_4/file_storage/all_experiments/`
(`<dataset>/<method>[/<config>]/<output_tree>[/<owner>]/...`), built by `scripts/consolidate_experiments.py`.
Canonical baselines: **ffatima** = Qwen-14B & Mistral; **ratirastogi** = Llama & DeepSeek; **rsenapati** =
Agentic RAG. See `docs/data_locations.md` and `scripts/all_experiments_README.md`.

## Documentation index

- `README.md` — setup + full workflow
- `docs/unity_setup.md` — Unity scratch + job submission
- `docs/gepa_prompt_optimization_plan.md`, `docs/gepa_flowchart.md` — GEPA subsystem
- `docs/data_locations.md`, `scripts/all_experiments_README.md` — where raw/consolidated data lives
- `docs/entity_extraction_comparison.md`, `docs/pipeline_comparison.md` — comparison vs `collapse-randomness-research`
- `docs/misinfo_error_compounding.md` — misinformation error-compounding experiment (HotpotQA); `scripts/hotpot-misinfo/`
- `visualization_outputs/README.md` — plots
