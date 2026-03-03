"""
uploader.py — Upload embedded chunks to Qdrant Cloud

Every point includes full source metadata so you can always trace
exactly where a piece of information came from:

    source_url          — exact URL or PDF URL the data came from
    source_type         — website | board_meeting | annual_report | tender_doc | pdf_document
    source_domain       — school domain (e.g. aucklandacademy.school.nz)
    source_page         — PDF page number (null for web pages)
    source_filename     — PDF filename if from a PDF (null for web pages)
    source_crawled_at   — ISO timestamp of when it was scraped
    source_school_name  — human-readable school name
    source_chunk_text   — the exact raw text this entity was extracted from
    source_label        — human-readable citation string for display / RAG
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from loguru import logger

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_RAW_COLLECTION, get_embedding_dim, RAW_TEXT_MAX_LENGTH
from chunker import Chunk


# ── Source metadata builder ───────────────────────────────────────────────────

def _source_metadata(chunk: Chunk) -> dict:
    m          = chunk.metadata
    source_url = m.get("source", "")
    parsed     = urlparse(source_url)
    is_pdf     = ".pdf" in source_url.lower()
    filename   = Path(parsed.path).name if is_pdf else None

    return {
        "source_url":         source_url,
        "source_type":        m.get("source_type", "unknown"),
        "source_domain":      m.get("domain", parsed.netloc),
        "source_school_name": m.get("school_name", ""),
        "source_page":        m.get("source_page"),
        "source_filename":    filename,
        "source_is_pdf":      is_pdf,
        "source_crawled_at":  datetime.now(timezone.utc).isoformat(),
        "source_chunk_text":  m.get("raw_text", "")[:RAW_TEXT_MAX_LENGTH],
        "source_label":       _source_label(
            school_name=m.get("school_name", ""),
            source_type=m.get("source_type", ""),
            source_url=source_url,
            filename=filename,
            page=m.get("source_page"),
        ),
    }


def _source_label(school_name, source_type, source_url, filename, page) -> str:
    labels = {
        "board_meeting": "Board Meeting",
        "annual_report": "Annual Report",
        "tender_doc":    "Tender Document",
        "project_doc":   "Project Document",
        "budget_page":   "Budget Page",
        "vendor_page":   "Vendor/Procurement Page",
        "website":       "Website",
        "pdf_document":  "PDF Document",
    }
    label = labels.get(source_type, source_type.replace("_", " ").title())
    if filename:
        loc = f"{filename}, p.{page}" if page else filename
        return f"{school_name} — {label} ({loc})"
    return f"{school_name} — {label} ({source_url[:80]})"


# ── Uploader ──────────────────────────────────────────────────────────────────

class QdrantUploader:

    def __init__(
        self,
        url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY,
        collection: str = QDRANT_COLLECTION,
    ):
        if not url:
            raise ValueError(
                "QDRANT_URL is not set.\n"
                "1. Go to https://cloud.qdrant.io and create a free cluster\n"
                "2. Add QDRANT_URL and QDRANT_API_KEY to your .env file"
            )
        if not api_key:
            raise ValueError("QDRANT_API_KEY is not set. Check your .env file.")

        try:
            from qdrant_client import QdrantClient
            self.client     = QdrantClient(url=url, api_key=api_key, timeout=60)
            self.collection = collection
            logger.info(f"Qdrant Cloud connected: {url} | collection: {collection}")
        except ImportError:
            raise ImportError("Run: pip install qdrant-client")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Qdrant Cloud at {url}\n"
                f"Error: {e}\n"
                "Check QDRANT_URL and QDRANT_API_KEY in your .env file."
            )

    def ensure_collection(self, vector_size: Optional[int] = None):
        from qdrant_client.models import Distance, VectorParams
        size     = vector_size or get_embedding_dim()
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )
            logger.success(f"Created collection: '{self.collection}' (dim={size})")
        else:
            logger.info(f"Collection '{self.collection}' already exists")

        # Ensure payload indexes for filterable fields
        from qdrant_client.models import PayloadSchemaType
        for field in ["metadata.type", "metadata.school_name"]:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # index already exists

    def upload(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """
        Upsert chunks into Qdrant Cloud.
        Safe to re-run — deterministic chunk IDs prevent duplicates.
        Each point payload = entity fields + full source tracing metadata.
        """
        from qdrant_client.models import PointStruct
        import uuid

        if not chunks:
            logger.warning("No chunks to upload")
            return 0

        missing = [c.chunk_id for c in chunks if c.vector is None]
        if missing:
            raise ValueError(f"{len(missing)} chunks have no vector. Run embedder first.")

        self.ensure_collection(vector_size=len(chunks[0].vector))

        def _to_uuid(hex_id: str) -> str:
            """Convert a hex chunk_id to a valid UUID for Qdrant."""
            # Use first 32 hex chars of the SHA256 hash to form a UUID
            h = hex_id[:32].ljust(32, "0")
            return str(uuid.UUID(h))

        uploaded = 0
        for i in range(0, len(chunks), batch_size):
            batch  = chunks[i:i + batch_size]
            points = [
                PointStruct(
                    id=_to_uuid(c.chunk_id),
                    vector=c.vector,
                    payload={
                        "page_content": c.embed_text,
                        "metadata": {
                            **c.metadata,
                            **_source_metadata(c),
                        },
                    },
                )
                for c in batch
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            uploaded += len(batch)
            logger.info(f"Uploaded {uploaded}/{len(chunks)} chunks")

        logger.success(f"✓ {uploaded} chunks stored in '{self.collection}'")
        return uploaded

    def search(
        self,
        query_vector: list[float],
        filter_conditions: Optional[dict] = None,
        limit: int = 10,
    ) -> list[dict]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qdrant_filter = None
        if filter_conditions:
            qdrant_filter = Filter(must=[
                FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
                for k, v in filter_conditions.items()
            ])
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
        )
        return [{**h.payload, "_score": h.score, "_id": h.id} for h in response.points]

    def filter_only(self, filter_conditions: dict, limit: int = 100) -> list[dict]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        results, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(must=[
                FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
                for k, v in filter_conditions.items()
            ]),
            limit=limit,
            with_payload=True,
        )
        return [h.payload for h in results]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection).count


# ── Raw text uploader ──────────────────────────────────────────────────────────

class RawTextUploader(QdrantUploader):
    """
    Uploads raw page text chunks to a separate Qdrant collection.
    Payload format: {page_content: text, metadata: {...}}
    Same as the entity uploader but for free-form text windows.
    """

    def __init__(
        self,
        url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY,
        collection: str = QDRANT_RAW_COLLECTION,
    ):
        super().__init__(url=url, api_key=api_key, collection=collection)

    def upload_raw(self, raw_chunks, batch_size: int = 64) -> int:
        """Upload RawChunk objects to the raw text collection."""
        from qdrant_client.models import PointStruct
        import uuid

        if not raw_chunks:
            logger.warning("No raw chunks to upload")
            return 0

        missing = [c.chunk_id for c in raw_chunks if c.vector is None]
        if missing:
            raise ValueError(f"{len(missing)} raw chunks have no vector.")

        self.ensure_collection(vector_size=len(raw_chunks[0].vector))

        def _to_uuid(hex_id: str) -> str:
            h = hex_id[:32].ljust(32, "0")
            return str(uuid.UUID(h))

        uploaded = 0
        for i in range(0, len(raw_chunks), batch_size):
            batch = raw_chunks[i:i + batch_size]
            points = [
                PointStruct(
                    id=_to_uuid(c.chunk_id),
                    vector=c.vector,
                    payload={
                        "page_content": c.text,
                        "metadata": c.metadata,
                    },
                )
                for c in batch
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            uploaded += len(batch)
            logger.info(f"Raw upload: {uploaded}/{len(raw_chunks)}")

        logger.success(f"✓ {uploaded} raw chunks stored in '{self.collection}'")
        return uploaded
