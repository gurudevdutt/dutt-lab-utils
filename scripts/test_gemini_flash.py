#!/usr/bin/env python3
"""One-off script: load .env, query Gemini Flash, print response. Run from repo root."""

import os
import sys
from pathlib import Path

# Repo root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Load .env into os.environ (no python-dotenv dependency)
env_file = ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                v = v.strip().strip('"').strip("'")
                os.environ[k.strip()] = v

from pittqlab_utils.llm.pittai import PittAIClient, PittAIModels

def main():
    client = PittAIClient()
    # Prefer Gemini Flash; fall back to Claude Haiku if model ID not available
    model = PittAIModels.GEMINI_FLASH
    try:
        print("Calling Gemini Flash...")
        resp = client.chat(
            "In one short sentence, what is quantum sensing?",
            model=model,
        )
    except ValueError as e:
        if "model identifier is invalid" in str(e) or "400" in str(e):
            print("Gemini Flash not available for this key, trying Claude Haiku...")
            model = PittAIModels.CLAUDE_HAIKU
            resp = client.chat(
                "In one short sentence, what is quantum sensing?",
                model=model,
            )
        else:
            raise
    print("Model:", resp.model)
    print("Response:", resp.text)
    print("Tokens:", resp.total_tokens)

if __name__ == "__main__":
    main()
