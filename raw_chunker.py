"""
raw_chunker.py — Split crawled page/PDF text into overlapping chunks
for the raw-text Qdrant collection.

Unlike the entity chunker (chunker.py) which stores structured extractions,
this stores the actual page text so RAG queries can find information that the
entity extractor missed (e.g. context, detail paragraphs, tables).

Each raw chunk carries:
    page_content  — the text window (what gets embedded)
    metadata      — school_name, domain, source_url, source_type, chunk_index, etc.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Any

from loguru import logger

from config import RAW_CHUNK_SIZE, RAW_CHUNK_OVERLAP


# ── Chunk dataclass (reused by embedder + uploader) ──────────────────────────

@dataclass
class RawChunk:
    chunk_id: str
    text: str                          # the text window
    metadata: dict[str, Any]
    vector: Optional[list[float]] = None


# ── Sliding-window splitter ──────────────────────────────────────────────────

def _split_text(text: str, size: int = RAW_CHUNK_SIZE,
                overlap: int = RAW_CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping windows of approximately `size` chars."""
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        window = text[start:end]
        chunks.append(window)
        start += size - overlap
    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def pages_to_raw_chunks(
    pages,           # list of spider.PageResult
    school_name: str,
    domain: str,
    min_length: int = 200,
) -> list[RawChunk]:
    """
    Convert crawled PageResult objects into overlapping RawChunks.

    Args:
        pages:       list of PageResult from the spider
        school_name: human-readable school name
        domain:      school domain
        min_length:  skip pages with less text than this
    """
    chunks: list[RawChunk] = []
    for page in pages:
        text = (page.text or "").strip()
        if len(text) < min_length:
            continue

        windows = _split_text(text)
        for idx, window in enumerate(windows):
            raw_id = hashlib.sha256(
                f"{school_name}::raw::{page.url}::{idx}::{window[:100]}".encode()
            ).hexdigest()

            chunks.append(RawChunk(
                chunk_id=raw_id,
                text=window.strip(),
                metadata={
                    "school_name": school_name,
                    "domain":      domain,
                    "source_url":  page.url,
                    "source_type": getattr(page, "source_type", "website"),
                    "title":       getattr(page, "title", ""),
                    "chunk_index": idx,
                    "total_chunks": len(windows),
                    "type":        "raw_text",
                },
            ))

    logger.info(f"Raw chunker: {len(pages)} pages → {len(chunks)} text chunks "
                f"(size={RAW_CHUNK_SIZE}, overlap={RAW_CHUNK_OVERLAP})")
    return chunks


def pdfs_to_raw_chunks(
    pdf_pages: list[dict],   # [{"url": ..., "page_number": ..., "text": ...}]
    school_name: str,
    domain: str,
    min_length: int = 50,
) -> list[RawChunk]:
    """
    Convert extracted PDF page texts into overlapping RawChunks.

    Each dict in pdf_pages should have keys: url, page_number, text, source_type.
    """
    chunks: list[RawChunk] = []
    for pp in pdf_pages:
        text = (pp.get("text") or "").strip()
        if len(text) < min_length:
            continue

        windows = _split_text(text)
        for idx, window in enumerate(windows):
            raw_id = hashlib.sha256(
                f"{school_name}::raw_pdf::{pp['url']}::p{pp.get('page_number', 0)}::{idx}::{window[:100]}".encode()
            ).hexdigest()

            chunks.append(RawChunk(
                chunk_id=raw_id,
                text=window.strip(),
                metadata={
                    "school_name":  school_name,
                    "domain":       domain,
                    "source_url":   pp["url"],
                    "source_type":  pp.get("source_type", "pdf_document"),
                    "source_page":  pp.get("page_number"),
                    "chunk_index":  idx,
                    "total_chunks": len(windows),
                    "type":         "raw_text",
                },
            ))

    logger.info(f"Raw PDF chunker: {len(pdf_pages)} PDF pages → {len(chunks)} text chunks")
    return chunks


def deduplicate_raw_chunks(chunks: list[RawChunk]) -> list[RawChunk]:
    """Remove chunks with identical text content for the same school."""
    seen = set()
    unique = []
    for c in chunks:
        key = (c.metadata.get("school_name", ""), c.text.strip().lower()[:200])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    removed = len(chunks) - len(unique)
    if removed:
        logger.info(f"Raw dedup: removed {removed} duplicate raw chunks")
    return unique
