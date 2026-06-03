"""
Write the best GEPA-optimized system prompt back to formatters.py.

Targets GEPA_RAG_GENERATION_SYSTEM_PROMPT, leaving _RAG_GENERATION_SYSTEM_PROMPT
(the original baseline) untouched.

Usage:
  # Pass the prompt text as a CLI argument
  python apply_best_prompt.py --prompt "Your optimized prompt here..."

  # Or read from stdin (useful for piping)
  echo "Your optimized prompt" | python apply_best_prompt.py --stdin
"""
import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
FORMATTERS_PATH = REPO_ROOT / "formatters.py"

PATTERN = re.compile(
    r'(GEPA_RAG_GENERATION_SYSTEM_PROMPT\s*=\s*""")(.*?)(""")',
    re.DOTALL,
)


def apply_prompt(new_prompt: str, formatters_path: Path = FORMATTERS_PATH) -> None:
    text = formatters_path.read_text(encoding="utf-8")

    if not PATTERN.search(text):
        print("ERROR: Could not find GEPA_RAG_GENERATION_SYSTEM_PROMPT in formatters.py")
        print("Make sure the variable uses triple-quote string syntax.")
        sys.exit(1)

    new_text = PATTERN.sub(
        lambda m: m.group(1) + new_prompt + m.group(3),
        text,
    )

    formatters_path.write_text(new_text, encoding="utf-8")
    print(f"Updated GEPA_RAG_GENERATION_SYSTEM_PROMPT in {formatters_path}")
    print("(_RAG_GENERATION_SYSTEM_PROMPT baseline is unchanged.)")
    print("\nNew prompt:")
    print("-" * 60)
    print(new_prompt)
    print("-" * 60)
    print("\nNext step: run scripts/eval.sh on the held-out test split to verify improvement.")


def main():
    parser = argparse.ArgumentParser(description="Apply optimized system prompt to formatters.py")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, help="Prompt text as a string argument")
    group.add_argument("--stdin", action="store_true", help="Read prompt text from stdin")
    args = parser.parse_args()

    if args.stdin:
        new_prompt = sys.stdin.read().strip()
    else:
        new_prompt = args.prompt.strip()

    if not new_prompt:
        print("ERROR: Prompt is empty.")
        sys.exit(1)

    apply_prompt(new_prompt)


if __name__ == "__main__":
    main()
