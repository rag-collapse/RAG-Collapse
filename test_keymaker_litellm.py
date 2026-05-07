"""
Verify that litellm routes calls through the keymaker proxy correctly.

Usage:
    python test_keymaker_litellm.py                    # test a small subset
    python test_keymaker_litellm.py --all              # test every model
    python test_keymaker_litellm.py --model openai/azure/gpt-5  # one model

Reads API_KEY from .env or the environment.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.environ.get("LITELLM_API_BASE", "https://thekeymaker.umass.edu/")
API_KEY  = os.environ.get("API_KEY")

if not API_KEY:
    print("ERROR: API_KEY not set. Add it to .env or export it.")
    sys.exit(1)

# All available keymaker models — prefixed with openai/ so litellm routes through proxy
ALL_MODELS = [
    "openai/gpt4o",
    "openai/gpt-5-mini",
    "openai/gpt5",
    "openai/deepseek-r1",
    "openai/deepseek-v3-0324",
    "openai/Phi-4-mini-reasoning",
    "openai/Phi-4-reasoning",
    "openai/claude-sonnet-3-7",
    "openai/mistral-large",
    "openai/claude-haiku-4-5",
    "openai/claude-opus-4-1",
    "openai/claude-sonnet-4-5",
    "openai/meta.llama3-3-70b",
    "openai/qwen3-coder-30b-a3b",
    "openai/qwen3-next-80b-a3b",
    "openai/gemma-3-12b-it",
    "openai/gemma-3-27b-it",
    "openai/gemma-3-4b-it",
]

# Quick subset — covers each provider family
DEFAULT_MODELS = [
    "openai/gpt4o",
    "openai/claude-haiku-4-5",
    "openai/gemma-3-12b-it",
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
        elapsed = time.time() - t0
        return True, text, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return False, str(e), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",   action="store_true", help="Test all models")
    parser.add_argument("--model", default=None,        help="Test one specific model")
    args = parser.parse_args()

    if args.model:
        models = [args.model]
    elif args.all:
        models = ALL_MODELS
    else:
        models = DEFAULT_MODELS

    litellm.suppress_debug_info = True
    litellm.drop_params = True  # gpt5 rejects temperature=0

    print(f"API base : {API_BASE}")
    print(f"Testing  : {len(models)} model(s)")
    print("=" * 70)

    passed, failed = 0, 0
    for model in models:
        ok, text, elapsed = test_model(model)
        status = "PASS" if ok else "FAIL"
        # Truncate long error messages
        display = text if len(text) < 80 else text[:77] + "..."
        print(f"  [{status}] {model:<50}  {elapsed:.1f}s  {display}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("=" * 70)
    print(f"  {passed} passed  |  {failed} failed")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
