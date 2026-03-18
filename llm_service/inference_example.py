import numpy as np
from open_source_llm import OpenSourceLLM, EmbeddingModel
from batch_examples import inference_example


def embedding_example(
    llm: OpenSourceLLM, embedding_model: EmbeddingModel
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
    embeddings = embedding_model.embed_batch(outputs)
    return embeddings


if __name__ == "__main__":
    # Reads VLLM_API_BASE from environment (set before running this script).
    # e.g. export VLLM_API_BASE="http://gypsum-gpu188.unity.rc.umass.edu:5150/v1"
    llm = OpenSourceLLM(
        model_name="qwen2.5-14b",
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
    )

    embedding_model = EmbeddingModel(
        "sentence-transformers/all-MiniLM-L6-v2", batch_size=8
    )

    try:
        print(llm)
        print(embedding_model)

        print("=== Inference Example ===")
        outputs = inference_example(llm)
        for i, output in enumerate(outputs):
            print(f"Output {i + 1}: {output}")

        print("=== Embedding Example ===")
        embeddings = embedding_example(llm, embedding_model)
        print("Embeddings shape:", embeddings.shape)
        print(
            "Similarity of 0-1:",
            embedding_model.similarity(embeddings[0], embeddings[1]),
        )
        print(
            "Similarity of 0-2:",
            embedding_model.similarity(embeddings[0], embeddings[2]),
        )

        print("Entire similarity matrix:")
        print(embedding_model.similarity_batch(embeddings, embeddings))

    finally:
        embedding_model.shutdown()
        print("\nCleaned up embedding model resources.")
