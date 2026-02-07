module load conda/latest
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ragenv
python -m pip install -r requirements.txt


export API_KEY= # Set your OpenAI API key here
export MODEL_MODE=api
export MODEL_NAME=openai/gpt4o
export DATASET_PATH=datasets/umass_data.entity.chatgpt.50.jsonl
export OUTPUT_PATH=example_experiments_output.json
export MAX_QUESTIONS=1
export NUM_ITERATIONS=5
export NUM_RUNS=10
export CHARS_PER_DOC=400

python -m pipeline.run_pipeline

# export MODEL_MODE=local
# export MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
# export DATASET_PATH=datasets/umass_data.entity.chatgpt.50.jsonl
# export OUTPUT_PATH=example_experiments_output.json
# export MAX_QUESTIONS=1
# export NUM_ITERATIONS=5
# export NUM_RUNS=10
# export CHARS_PER_DOC=400

# python -m pipeline.run_pipeline
