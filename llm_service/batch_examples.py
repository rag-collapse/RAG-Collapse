from common_llm import CommonLLM

def inference_example(llm: CommonLLM) -> str:
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