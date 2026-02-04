# set your API key in the .env file as API_KEY=your_key_here
from proprietary_llm import ProprietaryLLM

def inference_example(llm: ProprietaryLLM) -> str:
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

if __name__ == "__main__":
    llm = ProprietaryLLM(
        model_name="openai/gpt4o",
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
    )

    try:
        print(llm)

        print("=== Inference Example ===")
        outputs = inference_example(llm)
        for i, output in enumerate(outputs):
            print(f"Output {i + 1}: {output}")

    except Exception as e:
        print(f"An error occurred during LLM operations: {e}")