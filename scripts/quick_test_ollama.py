# quick_test_ollama.py — run from repo root: python scripts/quick_test_ollama.py
# Load .env so PortkeyBackend can find PITTAI_API_KEY (optional; Ollama needs no key).
import asyncio
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parent.parent
env_file = ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

from pittqlab_utils.llm.pittai import PittAIModels
from pittqlab_utils.llm import OllamaBackend, PortkeyBackend, ClassifierRouter

LABELS = ["email", "calendar", "code", "research", "chitchat", "reminder", "unknown"]

SAMPLES = [
    "Can you summarise my unread emails from today?",
    "What's on my calendar this afternoon?",
    "Refactor the studio_bridge to add a task queue",
    "Hey, good morning!",
    "Remind me to check the EPR data at 4pm",
]

async def main():
    ollama = OllamaBackend(model="llama3.2")
    portkey = PortkeyBackend(model=PittAIModels.CHEAP)
    router = ClassifierRouter([ollama, portkey], confidence_threshold=0.75)

    print("\n=== ping ===")
    await router.ping_all()
    for b in [ollama, portkey]:
        print(f"  {'✓' if b.is_available else '✗'} {b.name}")

    print("\n=== classify ===")
    for text in SAMPLES:
        r = await router.classify(text, labels=LABELS)
        print(f"  [{r.label:10s}] {r.confidence:.2f}  {r.backend:30s}  {text!r}")

    if hasattr(ollama, "aclose"):
        await ollama.aclose()

asyncio.run(main())