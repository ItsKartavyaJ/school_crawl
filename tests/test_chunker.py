"""
tests/test_chunker.py — Unit tests for chunking, deduplication, and metadata
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from extractor import ExtractedEntity
from chunker import entity_to_chunk, entities_to_chunks, deduplicate_chunks, Chunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_entity(
    entity_type="vendor",
    text="Oracle NZ Ltd for student software",
    source_url="https://school.nz/board/minutes",
    source_page=None,
    school_name="Test School",
    domain="school.nz",
    **attrs,
):
    default_attrs = {
        "vendor_name": "Oracle NZ Ltd",
        "service_type": "student software",
        "contract_value": "$48,500",
        "expiry_date": "30 June 2026",
        "status": "approved",
    }
    default_attrs.update(attrs)
    return ExtractedEntity(
        entity_type=entity_type,
        text=text,
        attributes=default_attrs,
        source_url=source_url,
        source_type="website",
        source_page=source_page,
        school_name=school_name,
        domain=domain,
    )


# ── Tests: entity_to_chunk ────────────────────────────────────────────────────

class TestEntityToChunk:
    def test_vendor_chunk_has_correct_metadata(self):
        entity = _make_entity()
        chunk = entity_to_chunk(entity)
        assert chunk is not None
        assert chunk.metadata["type"] == "vendor"
        assert chunk.metadata["school_name"] == "Test School"
        assert chunk.metadata["vendor_name"] == "Oracle NZ Ltd"
        assert chunk.metadata["contract_value"] == "$48,500"

    def test_vendor_embed_text_contains_key_info(self):
        entity = _make_entity()
        chunk = entity_to_chunk(entity)
        assert "Oracle NZ Ltd" in chunk.embed_text
        assert "Test School" in chunk.embed_text

    def test_budget_chunk(self):
        entity = _make_entity(
            entity_type="budget",
            text="Capital works $350,000",
            amount="350000", currency="NZD", category="capital works",
            funding_source="Ministry of Education", period="2024", status="approved",
        )
        chunk = entity_to_chunk(entity)
        assert chunk is not None
        assert chunk.metadata["type"] == "budget"
        assert chunk.metadata["amount"] == "350000"

    def test_unknown_entity_type_returns_none(self):
        entity = _make_entity(entity_type="unknown_thing")
        chunk = entity_to_chunk(entity)
        assert chunk is None

    def test_chunk_id_includes_source_url(self):
        """Chunk ID should differ for same text from different URLs."""
        e1 = _make_entity(source_url="https://school.nz/page1")
        e2 = _make_entity(source_url="https://school.nz/page2")
        c1 = entity_to_chunk(e1)
        c2 = entity_to_chunk(e2)
        assert c1.chunk_id != c2.chunk_id

    def test_chunk_id_includes_source_page(self):
        """Chunk ID should differ for same text from different PDF pages."""
        e1 = _make_entity(source_page=1)
        e2 = _make_entity(source_page=2)
        c1 = entity_to_chunk(e1)
        c2 = entity_to_chunk(e2)
        assert c1.chunk_id != c2.chunk_id

    def test_chunk_id_is_sha256_hex(self):
        entity = _make_entity()
        chunk = entity_to_chunk(entity)
        assert len(chunk.chunk_id) == 64  # SHA256 hex digest
        assert all(c in "0123456789abcdef" for c in chunk.chunk_id)

    def test_all_entity_types_produce_chunks(self):
        types_and_attrs = {
            "vendor": {"vendor_name": "X"},
            "budget": {"amount": "1000", "currency": "NZD", "category": "ops",
                       "funding_source": "", "period": "", "status": ""},
            "project": {"project_name": "X", "value": "", "timeline": "",
                        "status": "", "vendor": ""},
            "problem": {"description": "leak", "category": "infra",
                        "severity": "high", "date_mentioned": "", "resolution": ""},
            "board_member": {"name": "Jane", "role": "Chair",
                             "term_start": "2024", "term_end": "2027"},
            "contractor": {"contractor_name": "Bob", "trade": "plumbing",
                           "project": "fix", "contract_value": "", "expiry_date": ""},
        }
        for etype, attrs in types_and_attrs.items():
            entity = _make_entity(entity_type=etype, **attrs)
            chunk = entity_to_chunk(entity)
            assert chunk is not None, f"Failed for type: {etype}"
            assert chunk.metadata["type"] == etype


# ── Tests: entities_to_chunks ─────────────────────────────────────────────────

class TestEntitiesToChunks:
    def test_filters_unknown_types(self):
        entities = [
            _make_entity(entity_type="vendor"),
            _make_entity(entity_type="totally_fake"),
        ]
        chunks = entities_to_chunks(entities)
        assert len(chunks) == 1
        assert chunks[0].metadata["type"] == "vendor"

    def test_empty_input(self):
        assert entities_to_chunks([]) == []


# ── Tests: deduplicate_chunks ─────────────────────────────────────────────────

class TestDeduplicateChunks:
    def test_removes_exact_duplicates(self):
        entity = _make_entity()
        c1 = entity_to_chunk(entity)
        c2 = entity_to_chunk(entity)
        assert c1.chunk_id == c2.chunk_id
        result = deduplicate_chunks([c1, c2])
        assert len(result) == 1

    def test_keeps_different_chunks(self):
        e1 = _make_entity(source_url="https://school.nz/a")
        e2 = _make_entity(source_url="https://school.nz/b")
        c1 = entity_to_chunk(e1)
        c2 = entity_to_chunk(e2)
        result = deduplicate_chunks([c1, c2])
        assert len(result) == 2

    def test_empty_input(self):
        assert deduplicate_chunks([]) == []


# ── Tests: Chunk.to_qdrant_point ──────────────────────────────────────────────

class TestChunkQdrant:
    def test_to_qdrant_point_structure(self):
        entity = _make_entity()
        chunk = entity_to_chunk(entity)
        chunk.vector = [0.1, 0.2, 0.3]
        point = chunk.to_qdrant_point()
        assert point["id"] == chunk.chunk_id
        assert point["vector"] == [0.1, 0.2, 0.3]
        assert "school_name" in point["payload"]
