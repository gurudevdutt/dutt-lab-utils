#!/usr/bin/env python3
"""Quick test: load .env, call one model per provider to verify per-provider API keys work. Run from repo root.

Uses python-dotenv's load_dotenv() — the same pattern consuming apps should use before
creating PittAIClient (this package does not call load_dotenv itself).
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Load repo-root .env into os.environ (works regardless of current working directory)
load_dotenv(ROOT / ".env")

from pittqlab_utils.llm.pittai import PittAIClient, PittAIModels

PROMPT = "Reply with exactly: OK and the name of your model in one short sentence."


def main():
    client = PittAIClient()
    tests = [
        ("Google (Gemini Flash)", PittAIModels.GEMINI_FLASH),
        ("Anthropic (Claude Haiku)", PittAIModels.CLAUDE_HAIKU),
        ("OpenAI/Azure (GPT)", PittAIModels.GPT_5p1),
    ]
    for label, model in tests:
        print(f"\n--- {label} ---")
        key_source = client.get_api_key_source_for_model(model)
        print(f"Key read: {key_source}")
        try:
            resp = client.chat(PROMPT, model=model)
            print(f"Model: {resp.model}")
            print(f"Response: {resp.text[:200].strip()}...")
            print(f"Tokens: {resp.total_tokens}")
        except Exception as e:
            print(f"Error: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
