"""
tests/test_pdf_utils.py — Unit tests for PDF text extraction
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pdf_utils import PDFPage, PDFDocument


class TestPDFDocument:
    def test_non_empty_pages_filters_short_text(self):
        pages = [
            PDFPage(page_number=1, text="Short", char_count=5),
            PDFPage(page_number=2, text="A" * 100, char_count=100),
            PDFPage(page_number=3, text="   ", char_count=3),
        ]
        doc = PDFDocument(path="/test.pdf", filename="test.pdf", pages=pages, total_pages=3)
        non_empty = doc.non_empty_pages
        assert len(non_empty) == 1
        assert non_empty[0].page_number == 2

    def test_full_text_concatenation(self):
        pages = [
            PDFPage(page_number=1, text="Hello", char_count=5),
            PDFPage(page_number=2, text="World", char_count=5),
        ]
        doc = PDFDocument(path="/test.pdf", filename="test.pdf", pages=pages, total_pages=2)
        full = doc.full_text
        assert "Page 1" in full
        assert "Page 2" in full
        assert "Hello" in full
        assert "World" in full

    def test_empty_document(self):
        doc = PDFDocument(path="/test.pdf", filename="test.pdf", pages=[], total_pages=0)
        assert doc.non_empty_pages == []
        assert doc.full_text == ""
