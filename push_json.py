"""
push_json.py — Embed + upload an existing JSON export to Qdrant.

Usage:
    python push_json.py output/Aceacademycharter_20260301_222117.json
"""

import json
import sys
import time
from pathlib import Path

from loguru import logger

from chunker import Chunk
from embedder import get_embedder
from uploader import QdrantUploader


def main():
    if len(sys.argv) < 2:
        print("Usage: python push_json.py <path-to-json>")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"File not found: {json_path}")
        sys.exit(1)

    # ── Load JSON ──────────────────────────────────────────────────────────
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    logger.info(f"Loaded {len(records)} records from {json_path.name}")

    # ── Convert to Chunk objects ───────────────────────────────────────────
    chunks = []
    for rec in records:
        chunk_id   = rec.pop("chunk_id")
        embed_text = rec.pop("embed_text")
        # Everything else is metadata
        chunks.append(Chunk(
            chunk_id=chunk_id,
            embed_text=embed_text,
            metadata=rec,
            vector=None,
        ))

    # ── Embed ──────────────────────────────────────────────────────────────
    logger.info("Embedding chunks...")
    t0 = time.time()
    embedder = get_embedder()
    chunks = embedder.embed_chunks(chunks)
    logger.success(f"Embedding done in {time.time() - t0:.1f}s")

    # ── Upload to Qdrant ───────────────────────────────────────────────────
    logger.info("Uploading to Qdrant...")
    db = QdrantUploader()
    uploaded = db.upload(chunks)
    logger.success(f"Done — {uploaded} chunks pushed to Qdrant")


if __name__ == "__main__":
    main()
