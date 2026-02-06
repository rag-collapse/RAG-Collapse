#ROUND 1
#Convert AI responses to docs
#use AI docs as input
#evaluation
from formatters import (
    get_create_document_conversation, 
    get_rag_generation_conversation,
)
import json

def run_experiment(
    llm,
    model_name,
    num_iterations,
    num_runs_per_iteration,
    question,
) -> Dict:
{
  "experiment_metadata": {
    "model": model_name,
    "num_iterations": num_iterations,
    "num_runs_per_iteration": num_runs_per_iteration,
  },
  "questions": [
    {
      "question_id": 0,
      "question_text": question_text,
      "iterations": []
    }
  ]
}
for iteration in range(num_iterations):
    with open("datasets/umass_data.entity.chatgpt.50.jsonl") as f:
    sample_doc = json.loads(f.readline())

    context = get_context_str_from_docs(sample_doc["references"], chars_per_doc=400)
    rag_conversation = get_rag_generation_conversation(context=context, question=sample_doc["question"])

    print("Context:")
    print(context)

    print("\nRAG Generation Conversation:")
    for turn in rag_conversation:
        print(f"{turn['role'].upper()}: {turn['content']}\n")

    # and then to call with inference,
    batch = [rag_conversation] * 10
    outputs = llm.inference_batch(batch)

if __name__ == "__main__":
    llm = OpenSourceLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        gpu_memory_utilization=0.7,
        max_model_len=8192
    )
    output = run_experiment(
        llm,
        model_name = "Qwen/Qwen2.5-1.5B-Instruct",
        question = "Who are the most popular TikTok dancers?",
        num_iterations = 2,
        num_runs_per_iteration = 5,
    )


