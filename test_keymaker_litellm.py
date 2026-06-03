"""
Verify that litellm routes calls through the keymaker proxy correctly.

Approach (confirmed via Context7 litellm docs):
  Prefix model name with 'openai/' so litellm routes to the OpenAI-compatible
  endpoint at api_base, passing the rest as the model ID in the request body.

  Model IDs come from GET https://thekeymaker.umass.edu/v1/models
  (NOT the internal routing names like azure/gpt-4o or bedrock/...)

Usage:
    python test_keymaker_litellm.py                 # test GEPA models only
    python test_keymaker_litellm.py --all           # test every known model
    python test_keymaker_litellm.py --model openai/gpt4o

Reads API_KEY from .env or the environment.
"""
import argparse
import os
import sys
import time

import litellm
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.environ.get("LITELLM_API_BASE", "https://thekeymaker.umass.edu/")
API_KEY  = os.environ.get("API_KEY")

if not API_KEY:
    print("ERROR: API_KEY not set. Add it to .env or export it.")
    sys.exit(1)

# All model IDs from GET /v1/models — prefixed with 'openai/' for litellm routing
ALL_MODELS = [
    "openai/gpt4o",
    "openai/gpt-5-mini",
    "openai/gpt5",
    "openai/deepseek-r1",
    "openai/deepseek-v3-0324",
    "openai/Phi-4-mini-reasoning",
    "openai/Phi-4-reasoning",
    "openai/claude-sonnet-4-6",
    "openai/claude-haiku-4-5",
    "openai/claude-haiku-4-5-20251001",
    "openai/claude-opus-4-6",
    "openai/claude-opus-4-7",
    "openai/claude-sonnet-3-7",
    "openai/mistral-large",
    "openai/meta.llama3-3-70b",
    "openai/qwen3-coder-30b-a3b",
    "openai/qwen3-next-80b-a3b",
    "openai/gemma-3-12b-it",
    "openai/gemma-3-27b-it",
    "openai/gemma-3-4b-it",
    "openai/claude-opus-4-1",
    "openai/claude-sonnet-4-5",
]

# The 4 models used by the GEPA optimization pipeline
GEPA_MODELS = [
    "openai/claude-haiku-4-5",     # TASK_MODEL
    "openai/gemma-3-12b-it",       # DOC_GEN_MODEL
    "openai/gpt4o",                # JUDGE_MODEL
    "openai/claude-opus-4-1",      # REFLECTION_MODEL
]

MESSAGES = [{"role": "user", "content": "Reply with exactly: OK"}]


def test_model(model: str) -> tuple[bool, str, float]:
    t0 = time.time()
    try:
        resp = litellm.completion(
            model=model,
            messages=MESSAGES,
            api_base=API_BASE,
            api_key=API_KEY,
            temperature=0.0,
            max_tokens=16,
        )
        text = resp.choices[0].message.content.strip()
        return True, text, time.time() - t0
    except Exception as e:
        return False, str(e)[:120], time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",   action="store_true", help="Test all known models")
    parser.add_argument("--model", default=None,        help="Test one model e.g. openai/gpt4o")
    args = parser.parse_args()

    models = [args.model] if args.model else (ALL_MODELS if args.all else GEPA_MODELS)

    litellm.suppress_debug_info = True
    litellm.drop_params = True

    print(f"API base : {API_BASE}")
    print(f"Testing  : {len(models)} model(s)")
    print("=" * 78)

    passed, failed = 0, 0
    for model in models:
        ok, text, elapsed = test_model(model)
        status = "PASS" if ok else "FAIL"
        display = text if len(text) < 60 else text[:57] + "..."
        print(f"  [{status}] {model:<42}  {elapsed:5.1f}s  {display}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("=" * 78)
    print(f"  {passed} passed  |  {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
