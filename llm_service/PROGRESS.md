### Feb 4
- Created `proprietary_llm.py`, which stores the `ProprietaryLLM` class which I have made as close to the `OpenSourceLLM` class, excpet for the apply_chat_template method since that's handled internally. I also split it as a separate file to help readability
- Renamed `custom_llm.py` to `open_source_llm.py` to be more accurate
- If needed, we can create a `baseLLM` class
- Created a `litellm_example.py` file for testing. The batch example probably could be split into their own file since I reused them
- updated `requirements.txt` with litellm dependencies
- Note: Used AI for help in debugging some stuff

### Feb 3
- Created `inference_example.py`, that includes some basic usage
- Set up `requirements`, might need more updates
- Set up scripts and logs for unity

### Feb 2
- Created `llm_service`
- Created and implemented `OpenSourceLLM` class that includes necessary helpers to run local LLMs
