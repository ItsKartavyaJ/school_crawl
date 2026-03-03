"""
chunker.py — Convert extracted entities into Qdrant-ready chunks

Each chunk = one entity with:
  - embed_text: human-readable description (what gets embedded)
  - metadata:   all filterable fields + full source tracing
  - chunk_id:   deterministic hash (re-runs won't create duplicates)
"""

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Any
from collections import defaultdict

from loguru import logger
from rapidfuzz import fuzz

from extractor import ExtractedEntity
from config import RAW_TEXT_MAX_LENGTH


# ── Entity resolution — fuzzy vendor name canonicalisation ────────────────────

_FUZZY_THRESHOLD = 85  # fuzz.ratio score to consider two names the same


def _resolve_vendor_names(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    """
    Canonicalise vendor / contractor names across all entities.

    Groups vendor and contractor names by fuzzy similarity, picks the
    longest (most specific) name as canonical, and rewrites all others.
    This merges e.g. "Amazon", "Amazon Web Services", "amazon.com" into
    a single canonical name.
    """
    # Collect unique names from relevant entity types
    vendor_types = {"vendor", "contractor"}
    name_key_map = {"vendor": "vendor_name", "contractor": "contractor_name"}

    raw_names: list[str] = []
    for e in entities:
        etype = e.entity_type.lower()
        if etype in vendor_types:
            key = name_key_map[etype]
            name = e.attributes.get(key, e.text).strip()
            if name:
                raw_names.append(name)

    if not raw_names:
        return entities

    # Build canonical clusters via greedy matching
    canonical_map: dict[str, str] = {}  # raw_lower → canonical
    clusters: list[list[str]] = []

    for name in raw_names:
        nl = name.lower().strip()
        if nl in canonical_map:
            continue
        # Try to find a matching cluster
        matched = False
        for cluster in clusters:
            rep = cluster[0]
            if fuzz.ratio(nl, rep.lower()) >= _FUZZY_THRESHOLD:
                cluster.append(name)
                canonical_map[nl] = cluster[0]  # will be resolved below
                matched = True
                break
        if not matched:
            clusters.append([name])
            canonical_map[nl] = name

    # For each cluster, pick the longest name as canonical
    for cluster in clusters:
        canonical = max(cluster, key=len)
        for name in cluster:
            canonical_map[name.lower().strip()] = canonical

    # Rewrite attributes
    rewrites = 0
    for e in entities:
        etype = e.entity_type.lower()
        if etype in vendor_types:
            key = name_key_map[etype]
            old_name = e.attributes.get(key, e.text).strip()
            new_name = canonical_map.get(old_name.lower().strip(), old_name)
            if new_name != old_name:
                e.attributes[key] = new_name
                rewrites += 1

    if rewrites:
        n_clusters = len([c for c in clusters if len(c) > 1])
        logger.info(f"Entity resolution: {rewrites} name rewrites across {n_clusters} clusters")

    return entities


# ── Chunk ─────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    embed_text: str
    metadata: dict[str, Any]
    vector: Optional[list[float]] = None
    sparse_indices: Optional[list[int]] = None
    sparse_values: Optional[list[float]] = None

    def to_qdrant_point(self) -> dict:
        return {
            "id": self.chunk_id,
            "vector": self.vector,
            "payload": self.metadata,
        }


# ── embed_text builders (one per entity type) ─────────────────────────────────

def _vendor_text(e: ExtractedEntity) -> str:
    a = e.attributes
    parts = [f"Vendor: {a.get('vendor_name', e.text)}"]
    if a.get("service_type"):  parts.append(f"Service: {a['service_type']}")
    if a.get("contract_value"): parts.append(f"Value: {a['contract_value']}")
    if a.get("expiry_date"):   parts.append(f"Expiry: {a['expiry_date']}")
    if a.get("status"):        parts.append(f"Status: {a['status']}")
    parts.append(f"School: {e.school_name}")
    return ". ".join(parts)

def _budget_text(e: ExtractedEntity) -> str:
    a = e.attributes
    parts = [f"Budget: {a.get('category', 'general')}"]
    if a.get("amount"):         parts.append(f"Amount: {a.get('currency','')} {a['amount']}".strip())
    if a.get("funding_source"): parts.append(f"Funded by: {a['funding_source']}")
    if a.get("period"):         parts.append(f"Period: {a['period']}")
    if a.get("status"):         parts.append(f"Status: {a['status']}")
    parts.append(f"School: {e.school_name}")
    return ". ".join(parts)

def _project_text(e: ExtractedEntity) -> str:
    a = e.attributes
    parts = [f"Project: {a.get('project_name', e.text[:80])}"]
    if a.get("description"): parts.append(f"Description: {a['description']}")
    if a.get("value"):       parts.append(f"Value: {a['value']}")
    if a.get("timeline"):    parts.append(f"Timeline: {a['timeline']}")
    if a.get("status"):      parts.append(f"Status: {a['status']}")
    if a.get("vendor"):      parts.append(f"Vendor: {a['vendor']}")
    parts.append(f"School: {e.school_name}")
    return ". ".join(parts)

def _problem_text(e: ExtractedEntity) -> str:
    a = e.attributes
    parts = [f"Problem: {a.get('description', e.text[:120])}"]
    if a.get("category"):       parts.append(f"Category: {a['category']}")
    if a.get("severity"):       parts.append(f"Severity: {a['severity']}")
    if a.get("resolution"):     parts.append(f"Resolution: {a['resolution']}")
    if a.get("date_mentioned"): parts.append(f"Date: {a['date_mentioned']}")
    parts.append(f"School: {e.school_name}")
    return ". ".join(parts)

def _board_member_text(e: ExtractedEntity) -> str:
    a = e.attributes
    parts = [f"Board member: {a.get('name', e.text)}"]
    if a.get("role"):       parts.append(f"Role: {a['role']}")
    if a.get("term_start"): parts.append(f"Term start: {a['term_start']}")
    if a.get("term_end"):   parts.append(f"Term end: {a['term_end']}")
    parts.append(f"School: {e.school_name}")
    return ". ".join(parts)

def _contractor_text(e: ExtractedEntity) -> str:
    a = e.attributes
    parts = [f"Contractor: {a.get('contractor_name', e.text)}"]
    if a.get("trade"):          parts.append(f"Trade: {a['trade']}")
    if a.get("project"):        parts.append(f"Project: {a['project']}")
    if a.get("contract_value"): parts.append(f"Value: {a['contract_value']}")
    if a.get("expiry_date"):    parts.append(f"Expiry: {a['expiry_date']}")
    parts.append(f"School: {e.school_name}")
    return ". ".join(parts)


TEXT_BUILDERS = {
    "vendor":       _vendor_text,
    "budget":       _budget_text,
    "project":      _project_text,
    "problem":      _problem_text,
    "board_member": _board_member_text,
    "contractor":   _contractor_text,
}

# ── metadata builders ─────────────────────────────────────────────────────────

def _base_meta(e: ExtractedEntity) -> dict:
    return {
        "school_name": e.school_name,
        "domain":      e.domain,
        "type":        e.entity_type,
        "source":      e.source_url,
        "source_type": e.source_type,
        "source_page": e.source_page,
        "raw_text":    e.text[:RAW_TEXT_MAX_LENGTH],
    }

METADATA_BUILDERS = {
    "vendor": lambda e: {**_base_meta(e),
        "vendor_name":    e.attributes.get("vendor_name", ""),
        "service_type":   e.attributes.get("service_type", ""),
        "contract_value": e.attributes.get("contract_value", ""),
        "expiry_date":    e.attributes.get("expiry_date", ""),
        "status":         e.attributes.get("status", ""),
    },
    "budget": lambda e: {**_base_meta(e),
        "amount":         e.attributes.get("amount", ""),
        "currency":       e.attributes.get("currency", ""),
        "category":       e.attributes.get("category", ""),
        "period":         e.attributes.get("period", ""),
        "funding_source": e.attributes.get("funding_source", ""),
        "status":         e.attributes.get("status", ""),
    },
    "project": lambda e: {**_base_meta(e),
        "project_name": e.attributes.get("project_name", ""),
        "value":        e.attributes.get("value", ""),
        "timeline":     e.attributes.get("timeline", ""),
        "status":       e.attributes.get("status", ""),
        "vendor":       e.attributes.get("vendor", ""),
    },
    "problem": lambda e: {**_base_meta(e),
        "category":       e.attributes.get("category", ""),
        "severity":       e.attributes.get("severity", ""),
        "date_mentioned": e.attributes.get("date_mentioned", ""),
        "resolution":     e.attributes.get("resolution", ""),
    },
    "board_member": lambda e: {**_base_meta(e),
        "name":       e.attributes.get("name", ""),
        "role":       e.attributes.get("role", ""),
        "term_start": e.attributes.get("term_start", ""),
        "term_end":   e.attributes.get("term_end", ""),
    },
    "contractor": lambda e: {**_base_meta(e),
        "contractor_name": e.attributes.get("contractor_name", ""),
        "trade":           e.attributes.get("trade", ""),
        "project":         e.attributes.get("project", ""),
        "contract_value":  e.attributes.get("contract_value", ""),
        "expiry_date":     e.attributes.get("expiry_date", ""),
    },
}


# ── Public API ────────────────────────────────────────────────────────────────

def entity_to_chunk(entity: ExtractedEntity) -> Optional[Chunk]:
    etype = entity.entity_type.lower()
    text_fn = TEXT_BUILDERS.get(etype)
    meta_fn = METADATA_BUILDERS.get(etype)

    if not text_fn or not meta_fn:
        logger.warning(f"Unknown entity type: {etype}")
        return None

    embed_text = text_fn(entity)
    metadata   = meta_fn(entity)
    # Include source_url + source_page in hash to avoid cross-page collisions
    chunk_id   = hashlib.sha256(
        f"{entity.school_name}::{etype}::{entity.source_url}::{entity.source_page}::{entity.text[:200]}".encode()
    ).hexdigest()

    return Chunk(chunk_id=chunk_id, embed_text=embed_text, metadata=metadata)


def entities_to_chunks(entities: list[ExtractedEntity]) -> list[Chunk]:
    # Fuzzy-match vendor/contractor names before chunking
    entities = _resolve_vendor_names(entities)
    chunks  = [entity_to_chunk(e) for e in entities]
    valid   = [c for c in chunks if c is not None]
    skipped = len(chunks) - len(valid)
    if skipped:
        logger.warning(f"Skipped {skipped} entities with unknown type")
    logger.info(f"Produced {len(valid)} chunks from {len(entities)} entities")
    return valid


def deduplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """
    Content-aware deduplication.

    Two chunks are duplicates if they have the same (school, type, embed_text).
    When duplicates exist we keep the one with the richest metadata (most
    non-empty attribute values) so we don't lose detail.

    This prevents the same entity (e.g. "Amazon Smile") extracted from 28
    different pages from appearing 28 times while still keeping distinct
    entities that happen to share a source URL.
    """
    from collections import defaultdict

    buckets: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        key = (
            c.metadata.get("school_name", ""),
            c.metadata.get("type", ""),
            c.embed_text.strip().lower(),
        )
        buckets[str(key)].append(c)

    unique = []
    for group in buckets.values():
        # Pick the chunk with the most filled-in metadata fields
        best = max(group, key=lambda c: sum(
            1 for v in c.metadata.values() if v not in (None, "", 0)
        ))
        unique.append(best)

    removed = len(chunks) - len(unique)
    if removed:
        logger.info(f"Dedup: removed {removed} duplicate chunks "
                     f"({len(chunks)} → {len(unique)})")
    return unique
