import numpy as np
from sentence_transformers import SentenceTransformer
from open_source_llm import OpenSourceLLM


def inference_example(llm: OpenSourceLLM) -> str:
    message_batch = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Can you tell me a joke?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the mole number in chemistry?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "In economics, what is the law of supply and demand?",
            },
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "What is the type of error in programming when a variable is not defined?",
            },
        ],
    ]

    outputs = llm.inference_batch(message_batch)
    return outputs


def chat_template_example(llm: OpenSourceLLM) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you tell me a joke?"},
    ]
    prompt = llm.apply_chat_template(messages, add_generation_prompt=True)
    return prompt


def embedding_example(
    llm: OpenSourceLLM, embedding_model: SentenceTransformer
) -> np.ndarray:
    message_batch = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the best 5 video games of all time?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "What are the most popular 5 vide games in United States?",
            },
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "In economics, what is the law of supply and demand?",
            },
        ],
    ]

    outputs = llm.inference_batch(message_batch)
    embeddings = embedding_model.encode(outputs)
    return embeddings


if __name__ == "__main__":
    llm = OpenSourceLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        gpu_memory_utilization=0.7,
    )

    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    try:
        print(llm)
        print(embedding_model)

        print("=== Inference Example ===")
        outputs = inference_example(llm)
        for i, output in enumerate(outputs):
            print(f"Output {i + 1}: {output}")

        print("=== Chat Template Example ===")
        prompt = chat_template_example(llm)
        print("Prompt:", prompt)
        print(
            "PS: Any prompts to the LLMs gets parsed into a special format called chat template. It is a string not a list. Lists are used in APIs to handle conversations easier, this is abstracted in API calls, but not in local LLMs."
        )

        print("=== Embedding Example ===")
        embeddings = embedding_example(llm, embedding_model)
        print("Embeddings shape:", embeddings.shape)
        print(
            "Similarity of 0-1:",
            np.dot(embeddings[0], embeddings[1])
            / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])),
        )
        print(
            "Similarity of 0-2:",
            np.dot(embeddings[0], embeddings[2])
            / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[2])),
        )
    finally:
        # Clean up vLLM engine processes
        del llm.llm
        import gc
        gc.collect()
        print("\nCleaned up LLM resources.")


