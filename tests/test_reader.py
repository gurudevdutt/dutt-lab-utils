"""
tests/test_reader.py
Run with: pytest tests/ -v
"""
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from pittqlab_utils.pdf.reader import (
    extract_text,
    extract_text_batch,
    ExtractionResult,
    _is_sparse,
    SPARSE_CHARS_PER_PAGE,
)


# ── Unit tests for helpers ────────────────────────────────────────────────────

def test_is_sparse_with_empty_text():
    assert _is_sparse("", 5) is True

def test_is_sparse_with_dense_text():
    text = "a" * (SPARSE_CHARS_PER_PAGE * 10)  # well above threshold
    assert _is_sparse(text, 1) is False

def test_is_sparse_with_borderline_text():
    # exactly at threshold — should NOT be considered sparse
    text = "a" * (SPARSE_CHARS_PER_PAGE * 3)
    assert _is_sparse(text, 3) is False

def test_is_sparse_zero_pages():
    assert _is_sparse("some text", 0) is True


# ── Integration-style tests (mock the actual PDF/OCR calls) ──────────────────

@pytest.fixture
def dummy_pdf(tmp_path):
    """Create a dummy .pdf file (content doesn't matter, we mock the reader)."""
    p = tmp_path / "test.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def test_extract_text_uses_pymupdf_for_dense_pdf(dummy_pdf):
    dense_text = "word " * 500  # well above sparse threshold

    with patch("pittqlab_utils.pdf.reader._extract_with_pymupdf", return_value=(dense_text, 5)) as mock_mu:
        result = extract_text(dummy_pdf)

    mock_mu.assert_called_once()
    assert result.method == "pymupdf"
    assert result.pages == 5
    assert result.text == dense_text


def test_extract_text_falls_back_to_tesseract_for_sparse_pdf(dummy_pdf):
    sparse_text = "x"  # way below threshold
    ocr_text = "Recovered text via OCR " * 100

    with patch("pittqlab_utils.pdf.reader._extract_with_pymupdf", return_value=(sparse_text, 3)), \
         patch("pittqlab_utils.pdf.reader._extract_with_tesseract", return_value=(ocr_text, 3)) as mock_ocr:
        result = extract_text(dummy_pdf)

    mock_ocr.assert_called_once()
    assert result.method == "tesseract"
    assert result.text == ocr_text
    assert len(result.warnings) > 0  # should warn about fallback


def test_extract_text_force_ocr_skips_pymupdf(dummy_pdf):
    ocr_text = "OCR result"

    with patch("pittqlab_utils.pdf.reader._extract_with_pymupdf") as mock_mu, \
         patch("pittqlab_utils.pdf.reader._extract_with_tesseract", return_value=(ocr_text, 2)):
        result = extract_text(dummy_pdf, force_ocr=True)

    mock_mu.assert_not_called()
    assert result.method == "tesseract"


def test_extract_text_file_not_found():
    with pytest.raises(FileNotFoundError):
        extract_text(Path("nonexistent.pdf"))


def test_extract_text_wrong_extension(tmp_path):
    bad = tmp_path / "document.docx"
    bad.write_bytes(b"fake")
    with pytest.raises(ValueError):
        extract_text(bad)


def test_extraction_result_properties(dummy_pdf):
    with patch("pittqlab_utils.pdf.reader._extract_with_pymupdf", return_value=("hello world", 1)):
        result = extract_text(dummy_pdf)

    assert result.char_count == len("hello world")
    assert not result.is_empty


def test_extract_text_batch_skips_errors(tmp_path):
    good_pdf = tmp_path / "good.pdf"
    good_pdf.write_bytes(b"%PDF fake")

    with patch("pittqlab_utils.pdf.reader._extract_with_pymupdf", return_value=("text " * 200, 2)):
        results = extract_text_batch([good_pdf, Path("missing.pdf")], skip_errors=True)

    assert len(results) == 1  # missing file skipped, not raised
