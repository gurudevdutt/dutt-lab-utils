# dutt-lab-utils

Shared Python utilities for Dutt Lab projects.

## Install

In any project that needs this, add to `requirements.txt`:

```
dutt-lab-utils @ git+https://github.com/YOUR_USERNAME/dutt-lab-utils.git@main
```

Then install normally:
```bash
pip install -r requirements.txt
```

For **development** (if you're actively editing this package and want changes to reflect immediately):
```bash
git clone https://github.com/YOUR_USERNAME/dutt-lab-utils.git
cd dutt-lab-utils
pip install -e .
```

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

```python
from dutt_lab_utils.pdf import extract_text, extract_text_batch, ExtractionResult
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
