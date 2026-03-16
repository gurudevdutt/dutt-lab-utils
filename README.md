# dutt-lab-utils

Shared Python utilities for Dutt Lab projects (PDF extraction, Pitt AI Connect LLM client).

## Using from another repo

This package is **not on PyPI**. Install from a local clone or directly from GitHub:

```bash
# From a local path (editable — changes in dutt-lab-utils show up immediately)
pip install -e /path/to/dutt-lab-utils

# From GitHub (editable)
pip install -e git+https://github.com/gurudevdutt/dutt-lab-utils.git
```

In your code, import the **package name** `pittqlab_utils` (the name of the package under `src/`, not "dutt-lab-utils"):

```python
from pittqlab_utils.llm import PittAIClient, PittAIModels, PittAIResponse
from pittqlab_utils.pdf import extract_text, extract_text_batch, ExtractionResult
```

For the LLM client, set `PITTAI_API_KEY` in your environment or `.env` before calling (this library does not load `.env` itself). Optionally set `PITTAI_API_KEY_ANTHROPIC`, `PITTAI_API_KEY_GOOGLE`, and `PITTAI_API_KEY_OPENAI` for one key per provider; the client picks the key from the model string on each request (e.g. Claude → Anthropic key, Gemini → Google key, GPT → OpenAI key).

### Pinning in requirements.txt

To pin from Git in another project’s `requirements.txt`:

```
# Replace with your fork or org if needed
dutt-lab-utils @ git+https://github.com/gurudevdutt/dutt-lab-utils.git@main
```

Then: `pip install -r requirements.txt`. For local development on dutt-lab-utils, use `pip install -e /path/to/dutt-lab-utils` instead so edits apply immediately.

## System requirement: Tesseract

The PDF extractor falls back to Tesseract OCR for scanned PDFs. Install it on your OS:

```bash
# Mac
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr
```

pymupdf-only extraction works without Tesseract installed.

## Usage

### PDF extraction

```python
from pittqlab_utils.pdf import extract_text, extract_text_batch, ExtractionResult
from pathlib import Path

# Single PDF — auto-detects whether to use pymupdf or Tesseract
result = extract_text(Path("paper.pdf"))
print(result.text)
print(result.method)    # "pymupdf" or "tesseract"
print(result.pages)
print(result.warnings)  # any issues encountered

# Batch
results = extract_text_batch(list(Path("pdfs/").glob("*.pdf")))

# Force OCR (e.g. you know it's a scanned document)
result = extract_text(Path("scan.pdf"), force_ocr=True)
```

### Pitt AI Connect (LLM)

```python
from pittqlab_utils.llm import PittAIClient, PittAIModels

client = PittAIClient()  # uses PITTAI_API_KEY from environment
resp = client.chat("Summarize this.", model=PittAIModels.GEMINI_FLASH)
print(resp.text)

# Structured JSON
data = client.chat_json('Return JSON: {"score": 1-5, "reason": "..."} for this abstract.')
```

The client automatically sends `max_completion_tokens` instead of `max_tokens` for Azure/OpenAI models (e.g. GPT via Pitt AI Connect) so those backends work without extra configuration.
