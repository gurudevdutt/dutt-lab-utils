# dutt-lab-utils

## Purpose
Shared Python utilities for Dutt Lab projects. Currently contains PDF extraction
and Pitt AI Connect LLM client. Other shared code (common data models, etc.) can
be added here as the need arises across projects.

## Who uses this
- `research-paper-organizer` — PDF text extraction for born-digital papers
- `phd-applicant-scoring` — PDF extraction and Pitt AI Connect scoring
- `menakai` — Pitt AI Connect for multi-turn conversation and orchestration routing

## Architecture
```
src/
  dutt_lab_utils/
    pdf/
      reader.py         # unified pymupdf + tesseract extractor with auto-detection
    llm/
      pittai.py         # Pitt AI Connect client (Portkey gateway)
      __init__.py       # re-exports PittAIClient, PittAIModels, PittAIResponse
    __init__.py
```

## Key design decisions

### PDF extraction
- pymupdf is always tried first (fast, no system dependencies beyond pip)
- Tesseract is the fallback for scanned/image PDFs (requires system install)
- Auto-detection threshold: < 100 chars/page average → assume scanned
- `ExtractionResult` dataclass carries method used + warnings — callers can log or inspect these

### Pitt AI Connect LLM client
- Thin wrapper around the Portkey gateway using plain `requests` (no OpenAI SDK dependency)
- Follows the exact API format documented by Pitt AI Connect (x-portkey-api-key header,
  https://api.portkey.ai/v1/chat/completions endpoint)
- `PittAIModels` class holds all canonical model strings — update here when Pitt AI
  Connect changes model versions, and all consumer projects pick it up automatically
- `chat()` for text and multimodal calls; `chat_json()` for structured JSON output
- Retry with exponential backoff built in (3 attempts by default)
- API key read from `PITTAI_API_KEY` environment variable — consuming apps must call
  `load_dotenv()` in their own startup; this library does NOT call it
- Multimodal images encoded as base64 data URLs per Pitt AI Connect spec

## Adding new shared utilities
Before adding something here, ask: is this used by 2+ projects AND likely to diverge
if copied? If yes, add it. If it's only used by one project, keep it there.

## Coding conventions
- **Python 3.9+** — do NOT use 3.10+ union syntax (`X | Y`); use `Optional[X]` and
  `Union[X, Y]` from `typing` instead
- No side effects in extraction functions — pure input/output
- All new modules get a corresponding `__init__.py` that re-exports the public API
- Tests in `tests/` using pytest — run `pytest` from repo root
- Type hints required on all public functions

## Environment variables
This library reads from environment variables but does NOT call `load_dotenv()` itself.
Consuming projects are responsible for loading `.env` before calling library code.

```
PITTAI_API_KEY=...    # Pitt AI Connect key from Portkey portal
```

Copy `.env.example` (at repo root) to `.env` and fill in your key. Never commit `.env`.

## System dependencies (not pip-installable)
Tesseract must be installed on the OS:
- Mac: `brew install tesseract`
- Ubuntu/Debian: `sudo apt install tesseract-ocr`

Collaborators need this for OCR to work. pymupdf-only extraction works without it.

## Commands
**Always use a `.venv` and activate it** before installing or running tests:

```bash
# Create and activate venv (once)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install for development (editable — changes take effect immediately)
pip install -e .

# Run tests
pytest tests/ -v

# Run only integration tests (requires PITTAI_API_KEY in .env)
pytest tests/ -v -m integration

# Quick smoke test: query Gemini Flash (loads .env from repo root)
python scripts/test_gemini_flash.py
```

## What NOT to do
- Do NOT add project-specific logic here (paper metadata schemas, applicant rubrics, etc.)
- Do NOT import from research-paper-organizer or phd-applicant-scoring — this package
  has no knowledge of its consumers
- Do NOT change `ExtractionResult` or `PittAIResponse` field names without updating
  all consumer projects
- Do NOT require Tesseract at import time — it's imported lazily inside the fallback path
- Do NOT call `load_dotenv()` anywhere in this library
- Do NOT use Python 3.10+ syntax — keep everything compatible with 3.9+
