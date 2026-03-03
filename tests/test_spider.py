"""
tests/test_spider.py — Unit tests for URL scoring, selection, classification, and helpers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from spider import (
    SitemapURL,
    score_url,
    select_pages,
    classify_pdf,
    classify_page,
    _safe_pdf_filename,
    url_priority,
)


# ── Tests: score_url ──────────────────────────────────────────────────────────

class TestScoreUrl:
    def test_high_value_keyword_boosts_score(self):
        entry = SitemapURL(url="https://school.nz/board/minutes")
        scored = score_url(entry)
        assert scored.relevance_score > 0

    def test_low_value_keyword_reduces_score(self):
        entry = SitemapURL(url="https://school.nz/news/latest-blog-post")
        scored = score_url(entry)
        assert scored.relevance_score < 10

    def test_pdf_gets_bonus(self):
        pdf_entry = SitemapURL(url="https://school.nz/docs/report.pdf", is_pdf=True)
        html_entry = SitemapURL(url="https://school.nz/docs/report")
        pdf_scored = score_url(pdf_entry)
        html_scored = score_url(html_entry)
        assert pdf_scored.relevance_score > html_scored.relevance_score

    def test_deep_url_penalized(self):
        shallow = SitemapURL(url="https://school.nz/about")
        deep = SitemapURL(url="https://school.nz/a/b/c/d/e/f")
        s1 = score_url(shallow)
        s2 = score_url(deep)
        assert s1.relevance_score > s2.relevance_score

    def test_budget_pdf_high_score(self):
        entry = SitemapURL(url="https://school.nz/docs/budget-2024.pdf", is_pdf=True)
        scored = score_url(entry)
        # PDF bonus (20) + budget keyword (10) + pdf_bonus_keyword(5) + shallow depth
        assert scored.relevance_score >= 30


# ── Tests: select_pages ──────────────────────────────────────────────────────

class TestSelectPages:
    def test_respects_max_pages(self):
        entries = [SitemapURL(url=f"https://school.nz/page{i}") for i in range(50)]
        pages, pdfs = select_pages(entries, max_pages=10)
        assert len(pages) + len(pdfs) <= 10

    def test_separates_pdfs_from_pages(self):
        entries = [
            SitemapURL(url="https://school.nz/about"),
            SitemapURL(url="https://school.nz/docs/report.pdf", is_pdf=True),
        ]
        pages, pdfs = select_pages(entries, max_pages=10)
        assert all(not e.is_pdf for e in pages)
        assert all(e.is_pdf for e in pdfs)

    def test_pdf_budget_cap(self):
        entries = [SitemapURL(url=f"https://school.nz/doc{i}.pdf", is_pdf=True) for i in range(100)]
        pages, pdfs = select_pages(entries, max_pages=20)
        # PDF budget = 0.4 * 20 = 8
        assert len(pdfs) <= 8


# ── Tests: classify_pdf ──────────────────────────────────────────────────────

class TestClassifyPdf:
    def test_board_minutes(self):
        assert classify_pdf("https://school.nz/board-minutes.pdf", "board-minutes.pdf") == "board_meeting"

    def test_budget(self):
        assert classify_pdf("https://school.nz/budget-2024.pdf", "budget-2024.pdf") == "annual_report"

    def test_tender(self):
        assert classify_pdf("https://school.nz/tender-docs.pdf", "tender-docs.pdf") == "tender_doc"

    def test_project(self):
        assert classify_pdf("https://school.nz/capital-project.pdf", "capital-project.pdf") == "project_doc"

    def test_generic(self):
        assert classify_pdf("https://school.nz/info.pdf", "info.pdf") == "pdf_document"


# ── Tests: classify_page ──────────────────────────────────────────────────────

class TestClassifyPage:
    def test_board_meeting(self):
        assert classify_page("https://school.nz/minutes", "Board Meeting", "") == "board_meeting"

    def test_annual_report(self):
        assert classify_page("https://school.nz/report", "Annual Report 2024", "") == "annual_report"

    def test_budget(self):
        assert classify_page("https://school.nz/finance", "Budget Summary", "financial summary") == "budget_page"

    def test_vendor(self):
        assert classify_page("https://school.nz/procurement", "Procurement", "") == "vendor_page"

    def test_project(self):
        assert classify_page("https://school.nz/projects", "Capital Works", "") == "project_page"

    def test_generic(self):
        assert classify_page("https://school.nz/home", "Welcome", "Hello world") == "website"


# ── Tests: _safe_pdf_filename ─────────────────────────────────────────────────

class TestSafePdfFilename:
    def test_basic(self):
        name = _safe_pdf_filename("https://school.nz/docs/report.pdf")
        assert name.startswith("report_")
        assert name.endswith(".pdf")

    def test_collision_resistance(self):
        n1 = _safe_pdf_filename("https://school.nz/2023/report.pdf")
        n2 = _safe_pdf_filename("https://school.nz/2024/report.pdf")
        assert n1 != n2

    def test_no_extension(self):
        name = _safe_pdf_filename("https://school.nz/docs/download?id=123")
        assert name.endswith(".pdf")

    def test_query_string_stripped(self):
        name = _safe_pdf_filename("https://school.nz/docs/report.pdf?v=2")
        assert "?" not in name


# ── Tests: url_priority ──────────────────────────────────────────────────────

class TestUrlPriority:
    def test_priority_path_scores_positive(self):
        assert url_priority("https://school.nz/about") > 0
        assert url_priority("https://school.nz/board/governance") > 0

    def test_non_priority_scores_zero(self):
        assert url_priority("https://school.nz/random-page") == 0
