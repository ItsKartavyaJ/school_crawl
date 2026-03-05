"""
pdf_utils.py — Extract text from PDF documents page by page

Uses pymupdf (fitz) as primary, pdfplumber as fallback.
Falls back to OCR (Tesseract) for scanned/image-based PDFs.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class PDFPage:
    page_number: int    # 1-indexed
    text: str
    char_count: int


@dataclass
class PDFDocument:
    path: str
    filename: str
    pages: list[PDFPage]
    total_pages: int

    @property
    def full_text(self) -> str:
        return "\n".join(f"\n--- Page {p.page_number} ---\n{p.text}" for p in self.pages)

    @property
    def non_empty_pages(self) -> list[PDFPage]:
        return [p for p in self.pages if len(p.text.strip()) > 50]


def _ocr_pdf(path: str, filename: str) -> Optional[PDFDocument]:
    """
    OCR fallback for scanned/image PDFs using Tesseract.
    Requires: pip install pytesseract Pillow, plus Tesseract installed on system.
    """
    try:
        import fitz
        from PIL import Image
        import pytesseract
        import io
        import platform
        import shutil

        # Platform-aware Tesseract path
        if platform.system() == "Windows":
            default_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if Path(default_path).exists():
                pytesseract.pytesseract.tesseract_cmd = default_path
        elif not shutil.which("tesseract"):
            logger.warning("Tesseract not found on PATH — OCR unavailable")
            return None

        doc = fitz.open(path)
        pages = []
        for i, page in enumerate(doc):
            # Render page to image at 300 DPI for good OCR quality
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img).strip()
            if text:
                pages.append(PDFPage(page_number=i + 1, text=text, char_count=len(text)))
        doc.close()

        if pages:
            logger.info(f"OCR extracted {len(pages)} pages from {filename}")
            return PDFDocument(path=path, filename=filename, pages=pages, total_pages=len(pages))

        logger.warning(f"OCR produced no text for {filename}")
        return None

    except ImportError as e:
        logger.warning(f"OCR not available ({e}). Install: pip install pytesseract Pillow")
        return None
    except Exception as e:
        logger.warning(f"OCR failed on {filename}: {e}")
        return None


def extract_pdf(path: str) -> Optional[PDFDocument]:
    """Extract text from a PDF file, page by page. Returns None if unreadable."""
    path     = str(path)
    filename = Path(path).name

    # Try pymupdf first (fastest)
    try:
        import fitz
        doc   = fitz.open(path)
        pages = [
            PDFPage(page_number=i + 1, text=page.get_text("text").strip(),
                    char_count=len(page.get_text("text")))
            for i, page in enumerate(doc)
            if page.get_text("text").strip()
        ]
        doc.close()
        if pages:
            logger.info(f"Extracted {len(pages)} pages from {filename} via pymupdf")
            return PDFDocument(path=path, filename=filename, pages=pages, total_pages=len(pages))
        # No text found — try OCR before giving up
        logger.info(f"No text in {filename} (may be scanned) — trying OCR...")
        return _ocr_pdf(path, filename)
    except ImportError:
        logger.warning("pymupdf not installed, trying pdfplumber...")
    except Exception as e:
        logger.warning(f"pymupdf failed on {filename}: {e}")

    # Fallback: pdfplumber
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                # --- Table extraction: intercept structured tables ---
                table_text = ""
                try:
                    tables = page.extract_tables()
                    for table in (tables or []):
                        if table and len(table) > 1:  # header + at least 1 row
                            # Convert table rows to markdown-style text
                            header = [str(c or "").strip() for c in table[0]]
                            rows = []
                            for row in table[1:]:
                                cells = [str(c or "").strip() for c in row]
                                rows.append(" | ".join(cells))
                            table_text += " | ".join(header) + "\n"
                            table_text += " | ".join(["-" * len(h) for h in header]) + "\n"
                            table_text += "\n".join(rows) + "\n\n"
                except Exception:
                    pass  # table extraction failed, fall through to text

                text = page.extract_text() or ""
                combined = (text.strip() + "\n\n" + table_text.strip()).strip()
                if combined:
                    pages.append(PDFPage(page_number=i + 1, text=combined,
                                         char_count=len(combined)))
        if pages:
            logger.info(f"Extracted {len(pages)} pages from {filename} via pdfplumber")
            return PDFDocument(path=path, filename=filename, pages=pages, total_pages=len(pages))
        # No text found — try OCR before giving up
        logger.info(f"No text in {filename} via pdfplumber — trying OCR...")
        return _ocr_pdf(path, filename)
    except ImportError:
        logger.error("Neither pymupdf nor pdfplumber installed. Run: pip install pymupdf")
        return None
    except Exception as e:
        logger.error(f"pdfplumber also failed on {filename}: {e}")
        return None


def extract_pdf_first_page(path: str) -> str:
    """
    Lightweight first-page extractor for PDF relevance filtering.

    Returns extracted text from page 1 when possible; returns empty string on
    failure. Tries pymupdf, then pdfplumber, then OCR for page 1 only.
    """
    path = str(path)
    filename = Path(path).name

    # Fast path: pymupdf page 1 text
    try:
        import fitz
        doc = fitz.open(path)
        if len(doc) > 0:
            text = (doc[0].get_text("text") or "").strip()
            doc.close()
            if text:
                return text
        else:
            doc.close()
    except Exception:
        pass

    # Fallback: pdfplumber page 1 text
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            if pdf.pages:
                text = (pdf.pages[0].extract_text() or "").strip()
                if text:
                    return text
    except Exception:
        pass

    # Final fallback: OCR first page only
    try:
        import fitz
        from PIL import Image
        import pytesseract
        import io
        import platform
        import shutil

        if platform.system() == "Windows":
            default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if Path(default_path).exists():
                pytesseract.pytesseract.tesseract_cmd = default_path
        elif not shutil.which("tesseract"):
            return ""

        doc = fitz.open(path)
        if len(doc) == 0:
            doc.close()
            return ""
        pix = doc[0].get_pixmap(dpi=250)
        doc.close()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = (pytesseract.image_to_string(img) or "").strip()
        return text
    except Exception:
        logger.debug(f"First-page extraction failed for {filename}")
        return ""
