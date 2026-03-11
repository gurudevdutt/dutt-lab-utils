"""
dutt_lab_utils/pdf/reader.py

Unified PDF text extractor for Dutt Lab projects.

Strategy:
  1. Try pymupdf first (fast, works on born-digital PDFs)
  2. If extracted text is too sparse, the PDF is likely scanned —
     fall back to Tesseract OCR page by page

Usage:
    from dutt_lab_utils.pdf.reader import extract_text, ExtractionResult

    result = extract_text(Path("paper.pdf"))
    print(result.text)
    print(result.method)   # "pymupdf" or "tesseract"
    print(result.pages)    # number of pages processed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ── Tunable thresholds ────────────────────────────────────────────────────────

# If pymupdf extracts fewer than this many characters per page on average,
# we assume the PDF is scanned and switch to Tesseract.
SPARSE_CHARS_PER_PAGE = 100

# Tesseract language. Change to "eng+fra" etc. if needed.
TESSERACT_LANG = "eng"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    text: str
    method: Literal["pymupdf", "tesseract"]
    pages: int
    path: Path
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) == 0


# ── Core extraction functions ─────────────────────────────────────────────────

def _extract_with_pymupdf(pdf_path: Path) -> tuple[str, int]:
    """Extract text using pymupdf. Returns (text, page_count)."""
    import fitz  # pymupdf

    text_parts = []
    with fitz.open(str(pdf_path)) as doc:
        page_count = len(doc)
        for page in doc:
            text_parts.append(page.get_text())

    return "\n".join(text_parts), page_count


def _extract_with_tesseract(pdf_path: Path) -> tuple[str, int]:
    """
    Extract text using Tesseract OCR.
    Converts each PDF page to an image, then runs OCR.
    Requires: tesseract installed on the system (brew install tesseract / apt install tesseract-ocr)
    """
    import fitz
    import pytesseract
    from PIL import Image
    import io

    text_parts = []
    with fitz.open(str(pdf_path)) as doc:
        page_count = len(doc)
        for i, page in enumerate(doc):
            # Render page to image at 300 DPI for good OCR quality
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(img, lang=TESSERACT_LANG)
            text_parts.append(page_text)
            logger.debug(f"  OCR page {i+1}/{page_count}")

    return "\n".join(text_parts), page_count


def _is_sparse(text: str, page_count: int) -> bool:
    """Return True if the text looks like it came from a scanned/image PDF."""
    if page_count == 0:
        return True
    chars_per_page = len(text.strip()) / page_count
    return chars_per_page < SPARSE_CHARS_PER_PAGE


# ── Public API ────────────────────────────────────────────────────────────────

def extract_text(
    pdf_path: Path | str,
    force_ocr: bool = False,
    force_pymupdf: bool = False,
) -> ExtractionResult:
    """
    Extract text from a PDF, automatically choosing the best method.

    Args:
        pdf_path:      Path to the PDF file
        force_ocr:     Skip pymupdf, always use Tesseract (useful for known scans)
        force_pymupdf: Skip auto-detection, always use pymupdf (useful for speed)

    Returns:
        ExtractionResult with .text, .method, .pages, .warnings
    """
    pdf_path = Path(pdf_path)
    warnings = []

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.name}")

    # ── Step 1: try pymupdf ──
    if not force_ocr:
        try:
            text, pages = _extract_with_pymupdf(pdf_path)
        except Exception as e:
            logger.warning(f"pymupdf failed on {pdf_path.name}: {e}")
            warnings.append(f"pymupdf error: {e}")
            text, pages = "", 0

        if force_pymupdf:
            return ExtractionResult(
                text=text, method="pymupdf", pages=pages, path=pdf_path, warnings=warnings
            )

        # ── Step 2: check if result looks like a scanned PDF ──
        if not _is_sparse(text, pages):
            logger.info(f"{pdf_path.name}: extracted via pymupdf ({pages} pages)")
            return ExtractionResult(
                text=text, method="pymupdf", pages=pages, path=pdf_path, warnings=warnings
            )

        logger.info(f"{pdf_path.name}: sparse text ({len(text.strip())} chars, {pages} pages) — switching to OCR")
        warnings.append(f"pymupdf yielded sparse text; used Tesseract fallback")

    # ── Step 3: fall back to Tesseract ──
    try:
        text, pages = _extract_with_tesseract(pdf_path)
        logger.info(f"{pdf_path.name}: extracted via Tesseract OCR ({pages} pages)")
        return ExtractionResult(
            text=text, method="tesseract", pages=pages, path=pdf_path, warnings=warnings
        )
    except Exception as e:
        logger.error(f"Tesseract failed on {pdf_path.name}: {e}")
        warnings.append(f"Tesseract error: {e}")
        # Return whatever pymupdf gave us rather than crashing
        return ExtractionResult(
            text=text if not force_ocr else "",
            method="pymupdf",
            pages=pages,
            path=pdf_path,
            warnings=warnings,
        )


def extract_text_batch(
    pdf_paths: list[Path | str],
    force_ocr: bool = False,
    skip_errors: bool = True,
) -> list[ExtractionResult]:
    """
    Extract text from multiple PDFs.

    Args:
        pdf_paths:   List of PDF paths
        force_ocr:   Always use Tesseract
        skip_errors: If True, log errors and continue; if False, raise on first error

    Returns:
        List of ExtractionResult (same order as input)
    """
    results = []
    for i, path in enumerate(pdf_paths):
        try:
            result = extract_text(path, force_ocr=force_ocr)
            results.append(result)
            logger.info(f"[{i+1}/{len(pdf_paths)}] {Path(path).name}: {result.method}, {result.pages}p, {result.char_count} chars")
        except Exception as e:
            logger.error(f"[{i+1}/{len(pdf_paths)}] Failed on {path}: {e}")
            if not skip_errors:
                raise
    return results
