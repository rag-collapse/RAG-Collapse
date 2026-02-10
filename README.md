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
- `--model-mode`: Choose between `local` (local model) or `api` (API-based model like GPT-4)
- `--model-name`: Model identifier (e.g., `Qwen/Qwen2.5-1.5B-Instruct` for local or `openai/gpt4o` for API)
- `--dataset-path`: Path to your input dataset (JSONL format)
- `--output-path`: Where to save experiment results
- `--max-questions`: Maximum number of questions to process
- `--num-iterations`: Number of RAG iterations per run
- `--num-runs`: Number of experimental runs
- `--chars-per-doc`: Character limit per document

### Running on SLURM/HPC Clusters

The repository includes SLURM batch scripts in the [scripts/](scripts/) directory for running on HPC clusters.

#### Prerequisites
Before submitting jobs, ensure the logs directory exists:
```bash
mkdir -p logs
```

#### 1. Running the Full Pipeline ([pipeline_run.sh](scripts/pipeline_run.sh))

Submit the pipeline job to SLURM:
```bash
sbatch scripts/pipeline.sh
```

This script:
- Requests 1 GPU with 32GB memory
- Runs for up to 2 hours
- Activates the `ragenv` conda environment
- Executes the pipeline with your configured parameters

**SLURM Resources:**
- Job name: `evaluation`
- Time limit: 2 hours
- Partition: `gpu`
- GPU: 1 GPU (with VRAM constraints)
- Memory: 32GB
- CPUs: 2

You can modify the parameters in the script to use API mode or adjust experiment settings.

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
tail -f logs/pipeline_run_<job_id>.out
tail -f logs/pipeline_run_<job_id>.err

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
