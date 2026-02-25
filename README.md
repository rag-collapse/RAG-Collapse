# RAG-Collapsement-on-Self-Refined-Generation

## Setup

### Use Conda
1. Create a conda virtual environment:
   ```bash
   module load conda/latest
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
- Batch chat inference via the vLLM server
- Text embeddings and similarity calculations

Requires a running vLLM server. Start it first (see [Step 1](#step-1--start-the-vllm-server)), then:

```bash
export VLLM_API_BASE="http://<fqdn>:5150/v1"
python llm_service/inference_example.py
```

## Pipeline details

The pipeline studies **RAG collapse**: the model answers from retrieved context, then its answers are turned into “documents” and fed back as context for the next round.

We support three **document-setting variants** (set via `--pipeline-variant`):

- **Replace All** (`hybrid` with `--num-synth-docs 10 --num-db-docs 0`): Each round the model sees only *synthetic* docs (from its own prior outputs, expanded into web-style articles via a dedicated prompt). Round 0 uses the question’s references; from round 1 onward context is 10 synthetic docs. **10 rounds.**
- **Replace One** (`replace_one`): Start with the question’s references (up to 10). Each round, replace exactly one slot in that list with one new AI-generated doc; the list evolves over **20 rounds.**
- **Search** (`search`): Each round the model sees the top-k chunks from a vector store (chunked docs, embedded with SentenceTransformer by default). New generated docs are added to the store each round. **30 rounds.**

**Feedback step:** Each answer is passed through a “create document” prompt (see `formatters.py`) so the model produces a full web-style article; that text becomes the synthetic document(s) for the next iteration.

**Outputs:** One JSON file per run (e.g. `local_replace_all.json`, `local_replace_one.json`, `local_search.json`) with `experiment_metadata`, `questions`, and per-iteration `documents` and `runs`.

## Running the Pipeline

`--model-mode local` connects to a running **vLLM OpenAI-compatible server** instead of loading a model in-process. You must start that server and export its URL as `VLLM_API_BASE` before running any script that uses an LLM (`pipeline.py`, `evaluation.py`, `entity_extraction.py`, `llm_service/inference_example.py`).

### Step 1 — Start the vLLM Server

#### On SLURM/HPC (recommended)

Submit [scripts/start_llm_server.sh](scripts/start_llm_server.sh) as a separate job:

```bash
mkdir -p logs
sbatch scripts/start_llm_server.sh
```

The script starts a `vllm serve` process on port **5150** with tensor-parallel-size 2, serving `Qwen2.5-14B-Instruct` as `qwen2.5-14b`. It writes the reachable URL to the log file.

Wait for the server to be ready (look for `"Uvicorn running"` in the log):

```bash
# Replace <JOB_ID> with the ID returned by sbatch
tail -f logs/slurm-<JOB_ID>-vllm-qwen2.5-14b.log
```

Once ready, copy the **"Reachable at"** line from the log and export it:

```bash
export VLLM_API_BASE="http://<fqdn>:5150/v1"
# e.g. export VLLM_API_BASE="http://gypsum-gpu188.unity.rc.umass.edu:5150/v1"
```

#### Locally (single machine)

If you have a GPU locally, start the server in one terminal:

```bash
vllm serve <model-path-or-name> \
  --served-model-name qwen2.5-14b \
  --tensor-parallel-size 1 \
  --port 5150 \
  --host 0.0.0.0
```

Then in a second terminal:

```bash
export VLLM_API_BASE="http://localhost:5150/v1"
```

### Step 2 — Run the Pipeline

With `VLLM_API_BASE` set, run the pipeline directly:

```bash
mkdir -p experiment_outputs

python -u pipeline.py \
  --model-mode local \
  --model-name qwen2.5-14b \
  --dataset-path datasets/umass_data.entity.chatgpt.50.jsonl \
  --output-path experiment_outputs/output.json \
  --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 \
  --max-questions 8 \
  --num-runs 10 \
  --chars-per-doc 400
```

**Pipeline Parameters:**
- `--model-mode`: `local` (connects to vLLM server via `VLLM_API_BASE`) or `api` (e.g. GPT-4 via proxy)
- `--model-name`: Label used in output filenames; the server's served model name is auto-discovered
- `--dataset-path`: Input dataset (JSONL)
- `--output-path`: Where to save experiment results
- `--pipeline-variant`: **`hybrid`** (Replace All when used with 10 synth / 0 db), **`replace_one`** (one slot replaced per round), or **`search`**
- **Replace All** (hybrid, 10 rounds): use `--pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0`. Context each round = synthetic docs only.
- **Replace One** (20 rounds): use `--pipeline-variant replace_one`. Start with refs (up to 10); each round one slot is replaced with one new AI doc. Optional: `--stop-if-converged` to stop early when answers are stable for 4 consecutive rounds.
- **Search** (30 rounds): use `--pipeline-variant search`. Vector retrieval each round; embeddings default to **local** (SentenceTransformer, no API key). Use `--search-embedding-mode api` only if your API exposes embedding models.
- `--max-questions`: Limit number of questions (omit for all)
- `--max-iterations`: Cap on rounds for quick tests (e.g. `--max-iterations 2`)
- `--num-runs`: Responses per round (default 10)
- `--chars-per-doc`: Character limit per document

### Quick smoke test (1 question, 2 rounds)

```bash
mkdir -p experiment_outputs

# Replace All
python -u pipeline.py --model-mode local --model-name qwen2.5-14b \
  --pipeline-variant hybrid --num-synth-docs 10 --num-db-docs 0 \
  --dataset-path datasets/umass_data.entity.chatgpt.50.jsonl \
  --output-path experiment_outputs/test_replace_all.json --max-questions 1 --max-iterations 2

# Replace One
python -u pipeline.py --model-mode local --model-name qwen2.5-14b \
  --pipeline-variant replace_one \
  --dataset-path datasets/umass_data.entity.chatgpt.50.jsonl \
  --output-path experiment_outputs/test_replace_one.json --max-questions 1 --max-iterations 2

# Search (local SentenceTransformer embeddings; no API key needed)
python -u pipeline.py --model-mode local --model-name qwen2.5-14b \
  --pipeline-variant search --dataset-path datasets/umass_data.entity.chatgpt.50.jsonl \
  --output-path experiment_outputs/test_search.json --max-questions 1 --max-iterations 2
```

### Running on SLURM/HPC Clusters

The repository includes SLURM batch scripts in the [scripts/](scripts/) directory.

#### Prerequisites
```bash
mkdir -p logs experiment_outputs evaluation_outputs entity_extraction_output
```

#### 1. Start the vLLM Server ([start_llm_server.sh](scripts/start_llm_server.sh))

This must be done **before** submitting any LLM-dependent job.

```bash
sbatch scripts/start_llm_server.sh
# Note the job ID, e.g. 12345
```

**Server configuration (edit variables at the top of the script):**
- Model: `Qwen2.5-14B-Instruct` snapshot from `model_cache_dir`
- Served model name: `qwen2.5-14b`
- Port: `5150`
- Tensor parallel size: `2` (requires 2 GPUs with ≥40 GB VRAM each)
- Max concurrent sequences: `32`

Wait for the server to be ready, then get the URL:

```bash
tail -f logs/slurm-<JOB_ID>-vllm-qwen2.5-14b.log
# Look for: Reachable at: http://<fqdn>:5150/v1
```

Export the URL **before** submitting any downstream jobs:

```bash
export VLLM_API_BASE="http://<fqdn>:5150/v1"
```

#### 2. Running the Pipeline ([pipeline.sh](scripts/pipeline.sh))

```bash
sbatch --export=ALL scripts/pipeline.sh
```

`--export=ALL` forwards your current environment (including `VLLM_API_BASE`) into the job. The script runs the **three document-setting variants** in **local mode** with per-variant run counts (10 / 20 / 30) and writes:

```
experiment_outputs/qwen-7b/Qwen/Qwen2.5-7B-Instruct_local_replace_all.json
experiment_outputs/qwen-7b/Qwen/Qwen2.5-7B-Instruct_local_replace_one.json
experiment_outputs/qwen-7b/Qwen/Qwen2.5-7B-Instruct_local_search_test.json
```

(Pattern: `experiment_outputs/$OUTPUT_SUBDIR/$MODEL_local_<variant>.json`)

To use **API mode**: comment out the local block and uncomment the API block in the script; set `API_KEY` in your environment.

**Script variables (edit at top of pipeline.sh):**
- `MODEL` – model name used in output filenames (default: `Qwen/Qwen2.5-7B-Instruct`)
- `OUTPUT_SUBDIR` – subdirectory under `experiment_outputs` (default: `qwen-7b`); override with `OUTPUT_SUBDIR=my-run sbatch ...`
- `TPARALLEL` – tensor-parallel-size passed to the pipeline (default: `2`); should match the server's `--tensor-parallel-size`
- `EXTRA` – extra pipeline args (default: `--max-questions 50`); add `--max-iterations 2` for shorter test runs

**SLURM Resources:**
- Job name: `pipeline`
- Time limit: 24 hours
- Partition: `gpu`
- GPUs: 2 — constraint `vram40|vram48` with SM capability `sm_70` or later
- Memory: 80 GB
- CPUs: 4

#### 3. Running Evaluation ([eval.sh](scripts/eval.sh))

`evaluation.py` uses the vLLM server for LLM-based metrics. Ensure `VLLM_API_BASE` is set before submitting.

After pipeline completion:

```bash
sbatch --export=ALL scripts/eval.sh
```

Reads pipeline outputs and writes evaluation results:

```
# Reads from:
experiment_outputs/qwen-7b/Qwen/Qwen2.5-7B-Instruct_local_replace_all.json
experiment_outputs/qwen-7b/Qwen/Qwen2.5-7B-Instruct_local_replace_one.json
experiment_outputs/qwen-7b/Qwen/Qwen2.5-7B-Instruct_local_search_test.json

# Writes to:
evaluation_outputs/qwen-7b/Qwen/Qwen2.5-7B-Instruct_local_replace_all_eval.json
evaluation_outputs/qwen-7b/Qwen/Qwen2.5-7B-Instruct_local_replace_one_eval.json
evaluation_outputs/qwen-7b/Qwen/Qwen2.5-7B-Instruct_local_search_test_eval.json
```

(Pattern: `evaluation_outputs/$INPUT_SUBDIR/$MODEL_local_<variant>_eval.json`)

**Script variables (edit at top of eval.sh):**
- `INPUT_SUBDIR` – must match `OUTPUT_SUBDIR` from `pipeline.sh` (default: `qwen-7b`)
- `MODEL` – must match `MODEL` from `pipeline.sh` (default: `Qwen/Qwen2.5-7B-Instruct`)

**SLURM Resources:**
- Job name: `evaluation`
- Time limit: 8 hours
- Partition: `gpu,gpu-preempt`
- GPUs: 1 — constraint `vram40|vram48|vram80`
- Memory: 32 GB
- CPUs: 2

#### 4. Running Entity Extraction ([entity_extraction.sh](scripts/entity_extraction.sh))

`entity_extraction.py` uses the vLLM server. Ensure `VLLM_API_BASE` is set before submitting.

```bash
sbatch --export=ALL scripts/entity_extraction.sh
```

Uses `--model-mode local` with `Qwen/Qwen2.5-7B-Instruct` by default. Reads from hardcoded paths and writes to `entity_extraction_output/`:

```
# Reads from (hardcoded — update to match pipeline.sh output paths if needed):
experiment_outputs/local_search.json
experiment_outputs/local_replace_one.json
experiment_outputs/local_replace_all.json

# Writes to:
entity_extraction_output/
```

> **Note:** The input paths in `entity_extraction.sh` are hardcoded to `experiment_outputs/local_*.json` and do not automatically follow `pipeline.sh`'s `$OUTPUT_SUBDIR/$MODEL_local_*.json` output paths. Edit the `--experiment-files` list in the script to point at the actual pipeline outputs.

To use **API mode** instead: uncomment the API block and comment out the local block in the script.

**SLURM Resources:**
- Job name: `entity-extraction`
- Time limit: 7 hours
- Partition: `gpu`
- GPUs: 1 — constraint `vram40|vram48` with SM capability `sm_70` or later
- Memory: 24 GB
- CPUs: 2

#### 5. Running Inference Examples ([inference.sh](scripts/inference.sh))

Test the LLM service with:

```bash
sbatch --export=ALL scripts/inference.sh
```

Runs `llm_service/inference_example.py`, which demonstrates batch chat inference and embedding similarity using the running vLLM server. No GPU is allocated by the job itself — all GPU work is done by the server.

#### Monitoring SLURM Jobs

```bash
# Check job status
squeue --me

# Stream logs
tail -f logs/slurm-<JOB_ID>-vllm-qwen2.5-14b.log   # vLLM server
tail -f logs/pipeline_<JOB_ID>.out                   # pipeline
tail -f logs/evaluation_<JOB_ID>.out                 # evaluation
tail -f logs/entity-extraction_<JOB_ID>.out          # entity extraction
tail -f logs/inference_<JOB_ID>.out                  # inference example

# Cancel a job
scancel <JOB_ID>
```

## Example Experiment Pipeline
![Pipeline](example.png)
