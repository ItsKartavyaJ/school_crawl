"""
exporter.py — Export chunks to Excel and JSON
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger

from chunker import Chunk
from config import OUTPUT_DIR


COLUMN_ORDER = {
    "vendor":       ["school_name", "domain", "vendor_name", "service_type", "contract_value", "expiry_date", "status", "source_label", "source_type", "source_page", "raw_text"],
    "budget":       ["school_name", "domain", "category", "amount", "currency", "funding_source", "period", "status", "source_label", "source_type", "source_page", "raw_text"],
    "project":      ["school_name", "domain", "project_name", "value", "timeline", "status", "vendor", "source_label", "source_type", "source_page", "raw_text"],
    "problem":      ["school_name", "domain", "category", "severity", "date_mentioned", "resolution", "source_label", "source_type", "source_page", "raw_text"],
    "board_member": ["school_name", "domain", "name", "role", "term_start", "term_end", "source_label", "source_type", "source_page"],
    "contractor":   ["school_name", "domain", "contractor_name", "trade", "project", "contract_value", "expiry_date", "source_label", "source_type", "source_page", "raw_text"],
}

SHEET_NAMES = {
    "vendor": "Vendors", "budget": "Budgets", "project": "Projects",
    "problem": "Problems", "board_member": "Board Members", "contractor": "Contractors",
}


def export_excel(chunks: list[Chunk], school_name: str = "schools") -> str:
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = OUTPUT_DIR / f"{school_name.replace(' ', '_')[:40]}_{ts}.xlsx"

    by_type: dict[str, list[dict]] = {}
    for c in chunks:
        t = c.metadata.get("type", "unknown")
        by_type.setdefault(t, []).append(c.metadata)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        # Summary sheet
        pd.DataFrame({
            "Entity Type": [SHEET_NAMES.get(t, t) for t in by_type],
            "Count": [len(v) for v in by_type.values()],
        }).to_excel(writer, sheet_name="Summary", index=False)

        for etype, rows in by_type.items():
            df         = pd.DataFrame(rows)
            sheet_name = SHEET_NAMES.get(etype, etype.title())
            cols       = COLUMN_ORDER.get(etype, [])
            ordered    = [c for c in cols if c in df.columns]
            extra      = [c for c in df.columns if c not in ordered]
            df[ordered + extra].to_excel(writer, sheet_name=sheet_name, index=False)

            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(
                    max(len(str(cell.value or "")) for cell in col) + 4, 50
                )

    logger.success(f"Excel: {filepath}")
    return str(filepath)


def export_json(chunks: list[Chunk], school_name: str = "schools") -> str:
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = OUTPUT_DIR / f"{school_name.replace(' ', '_')[:40]}_{ts}.json"
    data     = [{"chunk_id": c.chunk_id, "embed_text": c.embed_text, **c.metadata} for c in chunks]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.success(f"JSON: {filepath}")
    return str(filepath)
