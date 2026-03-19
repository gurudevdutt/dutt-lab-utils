# AGENTS.md

## Cursor Cloud specific instructions

### Overview

`dutt-lab-utils` is a shared Python utility library (not a standalone application). It provides PDF extraction (`pittqlab_utils.pdf`) and LLM client integrations (`pittqlab_utils.llm`). See `CLAUDE.md` for full architecture and coding conventions.

### Running tests

```bash
source .venv/bin/activate
pytest tests/test_pittai.py tests/llm/ -v -m "not integration"
```

- `tests/test_reader.py` has a pre-existing import bug (`from dutt_lab_utils` should be `from pittqlab_utils`). Exclude it or expect a collection error.
- Integration tests (`pytest -m integration`) require `PITTAI_API_KEY` env var and make real API calls.
- Unit tests use mocks (`unittest.mock` and `respx`) — no external services needed.

### System dependency

Tesseract OCR must be installed at the OS level (`sudo apt install tesseract-ocr`). The VM snapshot includes it, but if missing, PDF OCR fallback tests will fail.

### Environment

- The venv lives at `.venv/`. Activate with `source .venv/bin/activate`.
- The package is installed in editable mode (`pip install -e ".[test]"`).
- `PITTAI_API_KEY` is only needed for integration tests and live scripts in `scripts/`; all unit tests work without it.

### Modules
- `pittqlab_utils.llm` — LLM backends (OllamaBackend, PortkeyBackend, ClassifierRouter)
- `pittqlab_utils.tools` — Gmail, Google Calendar, voice response (gTTS/ElevenLabs)
- `pittqlab_utils.pdf` — PDF extraction with Tesseract fallback

### Key constraints for agents
- tools/ has no knowledge of Telegram or menakaibot — pure utility only
- Email drafts are NEVER sent automatically — draft_email() only
- Voice notes must be .ogg Opus format for Telegram compatibility
