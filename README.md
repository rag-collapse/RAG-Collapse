# RAG-Collapsement-on-Self-Refined-Generation

## Setup

### Use Conda
1. Create a conda virtual environment:
   ```bash
   conda create -n ragenv python=3.11
   conda activate ragenv
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Running Examples

### LLM Service Examples
The `llm_service` directory contains example files demonstrating how to use the custom LLM and embedding models.

#### Running inference_example.py
This example demonstrates:
- Batch inference with a local LLM
- Chat template formatting
- Text embeddings and similarity calculations

To run:
```bash
python llm_service/inference_example.py
```

**Note**: The example uses `Qwen/Qwen2.5-1.5B-Instruct` model which will be downloaded automatically on first run. Make sure you have sufficient disk space and GPU memory available.

## Pipeline details

The pipeline studies **RAG collapse**: the model answers from retrieved context, then its answers are turned into “documents” and fed back as context for the next round.

We support three **document-setting variants** (set via `--pipeline-variant`):

- **Replace All** (`hybrid` with `--num-synth-docs 10 --num-db-docs 0`): Each round the model sees only *synthetic* docs (from its own prior outputs, expanded into web-style articles via a dedicated prompt). Round 0 uses the question’s references; from round 1 onward context is 10 synthetic docs. **10 rounds.**
- **Replace One** (`replace_one`): Start with the question’s references (up to 10). Each round, replace exactly one slot in that list with one new AI-generated doc; the list evolves over **20 rounds.**
- **Search** (`search`): Each round the model sees the top-k chunks from a vector store (chunked docs, embedded with SentenceTransformer by default). New generated docs are added to the store each round. **30 rounds.**

**Feedback step:** Each answer is passed through a “create document” prompt (see `formatters.py`) so the model produces a full web-style article; that text becomes the synthetic document(s) for the next iteration.

**Outputs:** One JSON file per run (e.g. `local_replace_all.json`, `local_replace_one.json`, `local_search.json`) with `experiment_metadata`, `questions`, and per-iteration `documents` and `runs`.

## Running the Pipeline

### Running Locally
You can run the pipeline directly with Python:

```bash
python -u pipeline.py \
  --model-mode local \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-path datasets/umass_data.entity.chatgpt.50.jsonl \
  --output-path experiment_outputs/output.json \
  --max-questions 8 \
  --num-iterations 2 \
  --num-runs 10 \
  --chars-per-doc 400
```

**Pipeline Parameters:**
- `--model-mode`: `local` (GPU) or `api` (e.g. GPT-4 via proxy)
- `--model-name`: Model identifier (e.g. `Qwen/Qwen2.5-1.5B-Instruct` or `openai/gpt4o`)
- `--dataset-path`: Input dataset (JSONL)
- `--output-path`: Where to save experiment results
- `--pipeline-variant`: **`hybrid`** (Replace All when used with 10 synth / 0 db), **`replace_one`** (document setting: one slot replaced per round), or **`search`**
- **Replace All** (hybrid, 10 rounds): use `--pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0`. Context each round = synthetic docs only (from model generations).
- **Replace One** (20 rounds): use `--pipeline-variant replace_one`. Start with refs (up to 10); each round one slot is replaced with one new AI doc. Optional: `--stop-if-converged` to stop early when answers are stable for 4 consecutive rounds.
- **Search** (30 rounds): use `--pipeline-variant search`. Vector retrieval each round; embeddings default to **local** (SentenceTransformer, no API key). Use `--search-embedding-mode api` only if your API exposes embedding models.
- `--max-questions`: Limit number of questions (omit for all)
- `--max-iterations`: Cap on rounds for quick tests (e.g. `--max-iterations 2`)
- `--num-runs`: Responses per round (default 10)
- `--chars-per-doc`: Character limit per document

### Testing variants

Quick smoke test (1 question, 2 rounds):

```bash
mkdir -p experiment_outputs

# Replace All (hybrid 10 synth, 0 refs)
python -u pipeline.py --model-mode local --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 \
  --dataset-path datasets/umass_data.entity.chatgpt.50.jsonl \
  --output-path experiment_outputs/test_replace_all.json --max-questions 1 --max-iterations 2

# Replace One (document setting: one slot replaced per round)
python -u pipeline.py --model-mode local --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --pipeline-variant replace_one \
  --dataset-path datasets/umass_data.entity.chatgpt.50.jsonl \
  --output-path experiment_outputs/test_replace_one.json --max-questions 1 --max-iterations 2

# Search (local embeddings; no API key needed)
python -u pipeline.py --model-mode local --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --pipeline-variant search --dataset-path datasets/umass_data.entity.chatgpt.50.jsonl \
  --output-path experiment_outputs/test_search.json --max-questions 1 --max-iterations 2
```

### Running on SLURM/HPC Clusters

The repository includes SLURM batch scripts in the [scripts/](scripts/) directory for running on HPC clusters.

#### Prerequisites
Before submitting jobs, ensure the logs directory exists:
```bash
mkdir -p logs
```

#### 1. Running the Pipeline ([pipeline.sh](scripts/pipeline.sh))

Submit the pipeline job to SLURM:
```bash
sbatch scripts/pipeline.sh
```

The script runs the **three document-setting variants** in **local mode** by default and writes:
- `experiment_outputs/local_replace_all.json` (Replace All: hybrid with `--num-synth-docs 10 --num-db-docs 0`)
- `experiment_outputs/local_replace_one.json` (Replace One: one slot replaced per round, 20 rounds)
- `experiment_outputs/local_search.json` (Search: vector retrieval; **local embeddings** by default, no API key)

To use **API mode**: in `scripts/pipeline.sh`, comment out the "Local mode" block and uncomment the "API mode" block; set `API_KEY` in your environment.

**Script variables (edit at top of pipeline.sh):**
- `DATASET` – input JSONL path (default: `datasets/umass_data.entity.chatgpt.50.jsonl`)
- `OUTDIR` – output directory (default: `experiment_outputs`)
- `COMMON` – shared args (e.g. `--num-runs 10`, `--chars-per-doc 400`)
- `EXTRA` – e.g. `--max-questions 50`; add `--max-iterations 2` for shorter test runs

**Environment:** The script sets `HF_HOME` and `HF_HUB_CACHE` to a local `model_cache` directory (avoids vLLM/HF cache errors). Search uses local SentenceTransformer embeddings by default; use `--search-embedding-mode api` only if your API provides embedding models.

**SLURM Resources:**
- Job name: `pipeline`
- Time limit: 2 hours
- Partition: `gpu`
- GPU: 1 GPU (with VRAM constraints)
- Memory: 32GB
- CPUs: 2

#### 2. Running Inference Examples ([inference_example.sh](scripts/inference_example.sh))

Test the LLM service with:
```bash
sbatch scripts/inference.sh
```

This script:
- Runs the inference example from `llm_service/inference_example.py`
- Loads CUDA 12.6 module
- Uses 24GB memory with 1 GPU
- Demonstrates batch inference, embeddings, and similarity calculations

**Note:** The script includes `nvidia-smi` for GPU diagnostics and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for better memory management.

#### 3. Running Evaluation ([eval.sh](scripts/eval.sh))

After pipeline completion, evaluate results with:
```bash
sbatch scripts/eval.sh
```

This script:
- Takes experiment outputs and generates evaluation metrics
- Default: reads from `experiment_outputs/test_output.json`
- Writes results to `evaluation_outputs/test_output.json`

You can modify the input/output paths in the script as needed.

#### Monitoring SLURM Jobs

Check your job status:
```bash
squeue --me
```

View job output logs:
```bash
# Pipeline logs
tail -f logs/pipeline_<job_id>.out
tail -f logs/pipeline_<job_id>.err

# Evaluation logs
tail -f logs/evaluation_<job_id>.out

# Inference example logs
tail -f logs/inference_example_<job_id>_<array_id>.out
```

Cancel a job:
```bash
scancel <job_id>
```

## Example Experiment Pipeline
![Pipeline](example.png)
