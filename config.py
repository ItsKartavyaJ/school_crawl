"""
config.py — Central configuration for School Intelligence Database
All settings loaded from .env file
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Gemini API ────────────────────────────────────────────────────────────────
# Used by both LangExtract (extraction) and the embedder — one key, no duplication
# Get your key from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
LANGEXTRACT_MODEL = os.getenv("LANGEXTRACT_MODEL", "gemini-2.5-flash")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

EMBEDDING_DIMS = {
    "text-embedding-004":       768,
    "embedding-001":            768,
    "text-embedding-3-small":   1536,
    "all-MiniLM-L6-v2":         384,
    "BAAI/bge-large-en-v1.5":   1024,
    "BAAI/bge-base-en-v1.5":    768,
    "all-mpnet-base-v2":        768,
}

def get_embedding_dim() -> int:
    return EMBEDDING_DIMS.get(EMBEDDING_MODEL, 768)

# ── Qdrant Cloud ──────────────────────────────────────────────────────────────
# Get cluster URL + API key from: https://cloud.qdrant.io
QDRANT_URL        = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "school_intel")
QDRANT_RAW_COLLECTION = os.getenv("QDRANT_RAW_COLLECTION", "school_intel_raw")

# ── Spider ────────────────────────────────────────────────────────────────────
# USE_JS_RENDERING: "auto" (default) = detect per-site, "true" = always, "false" = never
_js_raw = os.getenv("USE_JS_RENDERING", "auto").lower().strip()
if _js_raw in ("true", "1", "yes"):
    USE_JS_RENDERING = True      # always use JS rendering
elif _js_raw in ("false", "0", "no"):
    USE_JS_RENDERING = False     # never use JS rendering
else:
    USE_JS_RENDERING = "auto"    # auto-detect per site

RESPECT_ROBOTS_TXT   = os.getenv("RESPECT_ROBOTS_TXT", "true").lower() in ("true", "1", "yes")
MAX_PAGES_PER_SCHOOL = int(os.getenv("MAX_PAGES_PER_SCHOOL", 100))
CONCURRENT_REQUESTS  = int(os.getenv("CONCURRENT_REQUESTS", 5))
DOWNLOAD_DELAY       = float(os.getenv("DOWNLOAD_DELAY", 1.0))

PRIORITY_PATHS = os.getenv(
    "PRIORITY_PATHS",
    "about,team,board,staff,governance,vendors,contracts,budget,finance,"
    "projects,annual-report,minutes,meetings,tenders,procurement"
).split(",")

# Fraction of page budget reserved for PDFs (rest goes to HTML pages)
PDF_BUDGET_RATIO = float(os.getenv("PDF_BUDGET_RATIO", 0.4))

# PDF selection behavior:
# true  -> keep all discovered sitemap PDFs (no PDF budget cap)
# false -> apply PDF_BUDGET_RATIO cap
DOWNLOAD_ALL_PDFS = os.getenv("DOWNLOAD_ALL_PDFS", "true").lower() in ("true", "1", "yes")

# Allow downloading PDFs from these external hosts in addition to the school domain.
# Useful for CMS/CDN storage domains (e.g., resources.finalsite.net).
ALLOWED_EXTERNAL_PDF_HOSTS = [
    h.strip().lower()
    for h in os.getenv("ALLOWED_EXTERNAL_PDF_HOSTS", "resources.finalsite.net").split(",")
    if h.strip()
]

# Recency filter — drop extracted entities whose content dates are older
# than N days.  Especially targets budget/project entities referencing
# fiscal years from many years ago.  0 = disabled.  Default 365 (~1 year).
RECENCY_DAYS = int(os.getenv("RECENCY_DAYS", 365*2))

# ── Text extraction thresholds ────────────────────────────────────────────────
MIN_PAGE_TEXT_LENGTH = int(os.getenv("MIN_PAGE_TEXT_LENGTH", 200))
MIN_PDF_TEXT_LENGTH  = int(os.getenv("MIN_PDF_TEXT_LENGTH", 50))

# ── Retry settings ────────────────────────────────────────────────────────────
RETRY_ATTEMPTS     = int(os.getenv("RETRY_ATTEMPTS", 3))
RETRY_WAIT_MIN     = float(os.getenv("RETRY_WAIT_MIN", 1.0))   # seconds
RETRY_WAIT_MAX     = float(os.getenv("RETRY_WAIT_MAX", 10.0))  # seconds

# ── Raw text truncation (consistent across chunker + uploader) ────────────────
RAW_TEXT_MAX_LENGTH = int(os.getenv("RAW_TEXT_MAX_LENGTH", 1000))

# ── Raw text chunking (second-pass: full page text → overlapping chunks) ──────
RAW_CHUNK_SIZE    = int(os.getenv("RAW_CHUNK_SIZE", 1000))     # chars per chunk
RAW_CHUNK_OVERLAP = int(os.getenv("RAW_CHUNK_OVERLAP", 200))   # overlap between chunks
