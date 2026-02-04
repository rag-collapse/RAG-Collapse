# set your API key in the .env file as API_KEY=your_key_here
from proprietary_llm import ProprietaryLLM
from batch_examples import inference_example

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