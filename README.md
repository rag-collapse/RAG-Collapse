# RAG-Collapsement-on-Self-Refined-Generation

## Setup

### Option 1: Using Conda
1. Create a conda virtual environment:
   ```bash
   conda create -n rag-collapse python=3.10
   conda activate rag-collapse
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using venv
1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
2. Activate the virtual environment:
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
3. Install dependencies:
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

## Example Experiment Pipeline
![Pipeline](example.png)
