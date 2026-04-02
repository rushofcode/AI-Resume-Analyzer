# utils/pdf_parser.py
"""
PDF text extraction utility using pdfplumber (primary) with PyPDF2 as fallback.
pdfplumber handles tables and complex layouts better than most alternatives.
"""

import io
import logging
import pdfplumber
import PyPDF2

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_obj) -> str:
    """
    Extract plain text from an uploaded PDF file object.

    Strategy:
    1. Try pdfplumber first — preserves spacing and handles multi-column layouts.
    2. Fall back to PyPDF2 if pdfplumber fails or yields empty output.

    Args:
        file_obj: A file-like object (BytesIO or Streamlit UploadedFile).

    Returns:
        Extracted text as a single string, or empty string on total failure.
    """
    raw_bytes = file_obj.read() if hasattr(file_obj, "read") else file_obj

    # --- Attempt 1: pdfplumber ---
    try:
        text_parts = []
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        full_text = "\n".join(text_parts).strip()
        if full_text:
            logger.info("pdfplumber extraction successful (%d chars).", len(full_text))
            return full_text
    except Exception as exc:
        logger.warning("pdfplumber failed: %s — trying PyPDF2 fallback.", exc)

    # --- Attempt 2: PyPDF2 fallback ---
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
        pages = [reader.pages[i].extract_text() or "" for i in range(len(reader.pages))]
        full_text = "\n".join(pages).strip()
        if full_text:
            logger.info("PyPDF2 fallback successful (%d chars).", len(full_text))
            return full_text
    except Exception as exc:
        logger.error("PyPDF2 fallback also failed: %s", exc)

    return ""


def validate_pdf(file_obj) -> tuple[bool, str]:
    """
    Lightweight check that the uploaded file is a readable PDF.

    Returns:
        (is_valid, error_message)
    """
    raw_bytes = file_obj.read() if hasattr(file_obj, "read") else file_obj
    if not raw_bytes:
        return False, "Uploaded file is empty."
    if not raw_bytes.startswith(b"%PDF"):
        return False, "File does not appear to be a valid PDF."
    try:
        PyPDF2.PdfReader(io.BytesIO(raw_bytes))
        return True, ""
    except Exception as exc:
        return False, f"Could not open PDF: {exc}"
