# dutt-lab-utils
## Purpose
Shared Python utilities for Dutt Lab projects. Currently contains PDF extraction.
Other shared code (e.g. Portkey client wrappers, common data models) can be added here
as the need arises across projects.

## Who uses this
- `research-paper-organizer` — PDF text extraction for born-digital papers
- `phd-applicant-scoring` — PDF extraction for applicant documents (mix of scanned and digital)

## Architecture
```
src/
  pdf/
    reader.py       # unified pymupdf + tesseract extractor with auto-detection
```

## Key design decisions
- pymupdf is always tried first (fast, no system dependencies beyond pip)
- Tesseract is the fallback for scanned/image PDFs (requires system install)
- Auto-detection threshold: < 100 chars/page average → assume scanned
- `ExtractionResult` dataclass carries method used + warnings — callers can log or inspect these

## Adding new shared utilities
Before adding something here, ask: is this used by 2+ projects AND likely to diverge
if copied? If yes, add it. If it's only used by one project, keep it there.
Suggested future additions: `dutt_lab_utils/llm/` for shared Portkey client logic.

## Coding conventions
- Python 3.11+
- No side effects in extraction functions — pure input/output
- All new modules get a corresponding `__init__.py` that re-exports the public API
- Tests in `tests/` using pytest — run `pytest` from repo root
- Type hints required

## System dependencies (not pip-installable)
Tesseract must be installed on the OS:
- Mac: `brew install tesseract`
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
Collaborators need this for OCR to work. pymupdf-only extraction works without it.

## Commands
```bash
# Install for development (editable — changes take effect immediately)
pip install -e .

# Run tests
pytest tests/ -v
```

## What NOT to do
- Do NOT add project-specific logic here (paper metadata schemas, applicant rubrics, etc.)
- Do NOT import from research-paper-organizer or phd-applicant-scoring — this package
  has no knowledge of its consumers
- Do NOT change ExtractionResult field names without updating both consumer projects
- Do NOT require Tesseract at import time — it's imported lazily inside the fallback path
