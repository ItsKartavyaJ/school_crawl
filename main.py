"""
main.py — School Intelligence Database Pipeline

Usage:
    python main.py --url https://aucklandacademy.school.nz
    python main.py --url https://... --name "Auckland Academy"
    python main.py --csv schools.csv
    python main.py --url https://... --no-qdrant    # skip Qdrant, just export files
    python main.py --url https://... --resume        # resume from last completed step
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from loguru import logger

from config import OUTPUT_DIR, MIN_PAGE_TEXT_LENGTH, MIN_PDF_TEXT_LENGTH
from spider import crawl_school, SchoolCrawlResult
from pdf_utils import extract_pdf, PDFDocument
from extractor import get_extractor, ExtractionResult
from chunker import entities_to_chunks, deduplicate_chunks, Chunk
from raw_chunker import pages_to_raw_chunks, pdfs_to_raw_chunks, deduplicate_raw_chunks
from embedder import get_embedder, SparseVectorizer
from uploader import QdrantUploader, RawTextUploader
from exporter import export_excel, export_json


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _checkpoint_path(school_name: str, stage: str) -> Path:
    cp_dir = OUTPUT_DIR / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    safe_name = school_name.replace(" ", "_")[:40]
    return cp_dir / f"{safe_name}_{stage}.json"


def _save_checkpoint(data, school_name: str, stage: str):
    path = _checkpoint_path(school_name, stage)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    logger.debug(f"Checkpoint saved: {path}")


def _load_checkpoint(school_name: str, stage: str):
    path = _checkpoint_path(school_name, stage)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ── Step-level state machine ─────────────────────────────────────────────────

_PIPELINE_STEPS = [
    "crawl", "extract", "chunk", "embed", "upload", "raw_upload", "export"
]


def _state_path(school_name: str) -> Path:
    cp_dir = OUTPUT_DIR / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    safe = school_name.replace(" ", "_")[:40]
    return cp_dir / f"{safe}_pipeline_state.json"


def _load_state(school_name: str) -> dict:
    path = _state_path(school_name)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_steps": [], "school_name": school_name}


def _save_state(school_name: str, state: dict):
    path = _state_path(school_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, default=str)


def _step_done(state: dict, step: str) -> bool:
    return step in state.get("completed_steps", [])


def _mark_done(state: dict, step: str, school_name: str):
    if step not in state["completed_steps"]:
        state["completed_steps"].append(step)
    _save_state(school_name, state)


# ── Token budget estimator ───────────────────────────────────────────────────

_GEMINI_FLASH_CONTEXT = 1_000_000  # 1M token context for Gemini 2.5 Flash
_CHARS_PER_TOKEN = 4  # approximate for English text


def _estimate_token_budget(pages, pdfs) -> int:
    """Estimate total tokens across all pages + PDFs (rough: chars / 4)."""
    total_chars = sum(len(p.text) for p in pages)
    for pdf_info in pdfs:
        pdf_doc = extract_pdf(pdf_info.local_path)
        if pdf_doc:
            total_chars += sum(len(p.text) for p in pdf_doc.non_empty_pages)
    return total_chars // _CHARS_PER_TOKEN


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    url: str,
    school_name: str = "",
    upload_to_qdrant: bool = True,
    crawl_dir: Optional[str] = None,
    resume: bool = False,
) -> list[Chunk]:
    """
    Full pipeline for one school:
      1. Sitemap discovery → smart page selection
      2. Scrapling crawl of selected pages + PDF download
      3. LangExtract entity extraction (pages + PDFs) — parallelized
      4. Chunk entities
      5. Embed chunks (dense + sparse) — batched
      6. Upload to Qdrant Cloud (hybrid vectors)
      7. Export Excel + JSON

    Pass resume=True to skip already-completed steps (uses JSON state file).
    """

    pipeline_start = time.perf_counter()
    timings: dict[str, float] = {}

    logger.info("=" * 60)
    logger.info(f"Processing: {school_name or url}")
    logger.info("=" * 60)

    # Load resume state (or start fresh)
    state = _load_state(school_name or url) if resume else {"completed_steps": [], "school_name": school_name or url}
    if resume and state["completed_steps"]:
        logger.info(f"Resuming — completed steps: {state['completed_steps']}")

    extractor    = get_extractor()
    all_entities = []
    crawl_result = None
    chunks       = []
    raw_chunks   = []

    # ── Step 1 & 2: Crawl ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if resume and _step_done(state, "crawl"):
        logger.info("Step 1/7: Crawl — skipping (already done)")
        crawl_result = crawl_school(url=url, school_name=school_name, crawl_dir=crawl_dir)
        school_name = crawl_result.school_name
    else:
        logger.info("Step 1/7: Sitemap discovery + crawl...")
        crawl_result = crawl_school(url=url, school_name=school_name, crawl_dir=crawl_dir)
        school_name = crawl_result.school_name
        _mark_done(state, "crawl", school_name)
    domain = crawl_result.domain
    timings["crawl"] = time.perf_counter() - t0

    logger.info(
        f"Crawl complete: {len(crawl_result.pages)} pages, "
        f"{len(crawl_result.pdfs)} PDFs "
        f"(sitemap: {crawl_result.sitemap_urls_found} found, "
        f"{crawl_result.sitemap_urls_selected} selected) "
        f"[{timings['crawl']:.1f}s]"
    )

    # ── Token budget estimation ───────────────────────────────────────────────
    est_tokens = _estimate_token_budget(crawl_result.pages, crawl_result.pdfs)
    logger.info(f"Estimated extraction tokens: ~{est_tokens:,} "
                f"({'⚠ large corpus' if est_tokens > 900_000 else 'within budget'})")
    if est_tokens > 900_000:
        logger.warning(
            f"Token budget is high (~{est_tokens:,}). "
            f"Consider reducing MAX_PAGES or excluding large PDFs to control API costs."
        )

    # ── Step 3a: Extract from web pages (parallel) ────────────────────────────
    t0 = time.perf_counter()
    if resume and _step_done(state, "extract"):
        logger.info("Step 2/7: Extraction — loading from checkpoint")
        cp = _load_checkpoint(school_name, "entities")
        if cp:
            from extractor import ExtractedEntity
            all_entities = [
                ExtractedEntity(
                    entity_type=e["type"], text=e["text"], attributes=e["attrs"],
                    source_url=e["source"], source_type=e.get("source_type", "website"),
                    source_page=e.get("page"), school_name=school_name, domain=domain,
                )
                for e in cp
            ]
            logger.info(f"Loaded {len(all_entities)} entities from checkpoint")
    else:
        logger.info(f"Step 2/7: Extracting from {len(crawl_result.pages)} web pages (parallel)...")

        def _extract_page(page):
            if len(page.text.strip()) < MIN_PAGE_TEXT_LENGTH:
                return []
            result = extractor.extract_from_text(
                text=page.text, source_url=page.url, source_type=page.source_type,
                school_name=school_name, domain=domain,
            )
            return result.entities

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_extract_page, page): page for page in crawl_result.pages}
            for future in as_completed(futures):
                try:
                    all_entities.extend(future.result())
                except Exception as e:
                    page = futures[future]
                    logger.error(f"Extraction failed for {page.url}: {e}")

        # ── Step 3b: Extract from PDFs (parallel) ─────────────────────────────
        logger.info(f"Extracting from {len(crawl_result.pdfs)} PDFs (parallel)...")

        def _extract_pdf_pages(pdf_info):
            entities = []
            pdf_doc = extract_pdf(pdf_info.local_path)
            if not pdf_doc:
                return entities
            for page in pdf_doc.non_empty_pages:
                if len(page.text.strip()) < MIN_PDF_TEXT_LENGTH:
                    continue
                result = extractor.extract_from_text(
                    text=page.text, source_url=pdf_info.url, source_type=pdf_info.source_type,
                    school_name=school_name, domain=domain, source_page=page.page_number,
                )
                entities.extend(result.entities)
            return entities

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_extract_pdf_pages, pdf): pdf for pdf in crawl_result.pdfs}
            for future in as_completed(futures):
                try:
                    all_entities.extend(future.result())
                except Exception as e:
                    pdf = futures[future]
                    logger.error(f"PDF extraction failed for {pdf.url}: {e}")

        # Save extraction checkpoint
        _save_checkpoint(
            [{"type": e.entity_type, "text": e.text, "attrs": e.attributes,
              "source": e.source_url, "source_type": e.source_type,
              "page": e.source_page} for e in all_entities],
            school_name, "entities"
        )
        _mark_done(state, "extract", school_name)

    timings["extraction"] = time.perf_counter() - t0
    logger.info(f"Total entities extracted: {len(all_entities)} [{timings['extraction']:.1f}s]")

    # ── Step 4: Chunk ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("Step 3/7: Chunking...")
    chunks = deduplicate_chunks(entities_to_chunks(all_entities))
    timings["chunking"] = time.perf_counter() - t0
    logger.info(f"Unique chunks: {len(chunks)} [{timings['chunking']:.1f}s]")
    _mark_done(state, "chunk", school_name)

    if not chunks:
        logger.warning("No chunks produced — check extraction step.")
        return []

    # ── Step 5: Embed ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if resume and _step_done(state, "embed"):
        logger.info("Step 4/7: Embedding — skipping (already done)")
    else:
        logger.info("Step 4/7: Embedding entities (dense + sparse)...")
        chunks = get_embedder().embed_chunks(chunks)
        _mark_done(state, "embed", school_name)
    timings["embedding"] = time.perf_counter() - t0
    logger.info(f"Embedding complete [{timings['embedding']:.1f}s]")

    # ── Step 6: Upload ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if upload_to_qdrant:
        if resume and _step_done(state, "upload"):
            logger.info("Step 5/7: Upload — skipping (already done)")
        else:
            logger.info("Step 5/7: Uploading entities to Qdrant Cloud...")
            db = QdrantUploader()
            db.upload(chunks)
            logger.success(f"Qdrant entity points: {db.count()}")
            _mark_done(state, "upload", school_name)
    else:
        logger.info("Step 5/7: Skipping Qdrant (--no-qdrant)")
    timings["upload"] = time.perf_counter() - t0

    # ── Step 6b: Raw text chunks → separate collection ────────────────────────
    t0 = time.perf_counter()
    if resume and _step_done(state, "raw_upload"):
        logger.info("Step 6/7: Raw upload — skipping (already done)")
    else:
        logger.info("Step 6/7: Building raw text chunks...")
        embedder = get_embedder()

        # Raw chunks from web pages
        raw_chunks = pages_to_raw_chunks(
            crawl_result.pages, school_name=school_name, domain=domain,
            min_length=MIN_PAGE_TEXT_LENGTH,
        )

        # Raw chunks from PDFs
        pdf_page_dicts = []
        for pdf_info in crawl_result.pdfs:
            pdf_doc = extract_pdf(pdf_info.local_path)
            if pdf_doc:
                for page in pdf_doc.non_empty_pages:
                    pdf_page_dicts.append({
                        "url": pdf_info.url,
                        "page_number": page.page_number,
                        "text": page.text,
                        "source_type": pdf_info.source_type,
                    })
        raw_chunks.extend(pdfs_to_raw_chunks(
            pdf_page_dicts, school_name=school_name, domain=domain,
            min_length=MIN_PDF_TEXT_LENGTH,
        ))

        raw_chunks = deduplicate_raw_chunks(raw_chunks)
        logger.info(f"Raw text chunks after dedup: {len(raw_chunks)}")

        if raw_chunks:
            # Embed raw chunks — dense vectors
            texts = [c.text for c in raw_chunks]
            vectors = embedder.embed(texts)
            for chunk, vec in zip(raw_chunks, vectors):
                chunk.vector = vec

            # Sparse vectors for raw chunks
            sparse_vecs = SparseVectorizer.vectorize_batch(texts)
            for chunk, (indices, values) in zip(raw_chunks, sparse_vecs):
                chunk.sparse_indices = indices
                chunk.sparse_values = values

            if upload_to_qdrant:
                logger.info("Uploading raw text chunks to Qdrant...")
                raw_db = RawTextUploader()
                raw_db.upload_raw(raw_chunks)
                logger.success(f"Qdrant raw text points: {raw_db.count()}")

        _mark_done(state, "raw_upload", school_name)
    timings["raw_upload"] = time.perf_counter() - t0

    # ── Export ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    excel_path = export_excel(chunks, school_name)
    json_path  = export_json(chunks, school_name)
    timings["export"] = time.perf_counter() - t0
    _mark_done(state, "export", school_name)
    logger.success(f"Excel: {excel_path}")
    logger.success(f"JSON:  {json_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.perf_counter() - pipeline_start
    logger.info(f"\n{'─' * 50}")
    logger.info(f"Pipeline complete for {school_name} in {total:.1f}s")
    for stage, duration in timings.items():
        logger.info(f"  {stage:15s}: {duration:6.1f}s")
    logger.info(f"  {'TOTAL':15s}: {total:6.1f}s")
    logger.info(f"  Entities: {len(all_entities)} → Chunks: {len(chunks)}")
    logger.info(f"  Raw text chunks: {len(raw_chunks)}")
    logger.info(f"{'─' * 50}")

    return chunks


def run_from_csv(csv_path: str, upload_to_qdrant: bool = True):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "url" not in df.columns:
        raise ValueError("CSV must have a 'url' column")
    name_col   = next((c for c in ["name", "school_name"] if c in df.columns), None)
    all_chunks = []
    for _, row in df.iterrows():
        chunks = run_pipeline(
            url=row["url"],
            school_name=row[name_col] if name_col else "",
            upload_to_qdrant=upload_to_qdrant,
        )
        all_chunks.extend(chunks)
    if all_chunks:
        export_excel(all_chunks, "all_schools")
        export_json(all_chunks, "all_schools")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="School Intelligence Database")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",  help="School website URL")
    group.add_argument("--csv",  help="CSV file with school URLs")
    parser.add_argument("--name",       default="", help="School name (optional with --url)")
    parser.add_argument("--no-qdrant",  action="store_true", help="Skip Qdrant upload")
    parser.add_argument("--crawl-dir",  default=None, help="Checkpoint dir for pause/resume")
    parser.add_argument("--resume",     action="store_true", help="Resume from last completed step")
    args = parser.parse_args()

    if args.csv:
        run_from_csv(args.csv, upload_to_qdrant=not args.no_qdrant)
    else:
        run_pipeline(
            url=args.url,
            school_name=args.name,
            upload_to_qdrant=not args.no_qdrant,
            crawl_dir=args.crawl_dir,
            resume=args.resume,
        )
