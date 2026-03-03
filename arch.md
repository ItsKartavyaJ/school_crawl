# School Intelligence Crawler — Architecture

## What This Project Does

This is a **school website intelligence pipeline**. Given a school's URL, it:

1. **Crawls** the website (HTML pages + PDFs)
2. **Extracts** structured entities (vendors, budgets, projects, board members, etc.) using an LLM
3. **Embeds** everything into vectors
4. **Uploads** to two Qdrant vector database collections
5. **Exports** to Excel/JSON for offline use
6. **Queries** via CLI or Jupyter notebook (semantic search + RAG)

The goal is to build a searchable database of school procurement data — who are their vendors, what contracts do they have, what projects are underway, etc.

---

## Pipeline Flow (7 Steps)

```
                    School URL
                        │
                        ▼
            ┌───────────────────────┐
            │  Step 1: CRAWL        │  spider.py
            │  sitemap discovery    │
            │  page selection       │
            │  HTML scrape + PDF    │
            └───────┬───────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   HTML pages             PDF files
   (PageResult)           (PDFResult)
          │                   │
          │    ┌──────────────┘
          │    │  pdf_utils.py extracts
          │    │  text from each PDF page
          ▼    ▼
            ┌───────────────────────┐
            │  Step 2: EXTRACT      │  extractor.py
            │  LangExtract + Gemini │
            │  parallel (4 workers) │
            └───────┬───────────────┘
                    │
                    ▼
          ExtractedEntity[]
          (vendor, budget, project,
           problem, board_member,
           contractor)
                    │
          ┌─────────┴─────────────────────┐
          ▼                               ▼
  ┌─────────────────┐          ┌─────────────────────┐
  │ Step 3: CHUNK   │          │ Step 6: RAW CHUNK   │
  │ chunker.py      │          │ raw_chunker.py      │
  │ entity → Chunk  │          │ sliding window      │
  │ + dedup         │          │ 1000 char / 200 ovlp│
  └────────┬────────┘          └──────────┬──────────┘
           │                              │
           ▼                              ▼
  ┌─────────────────┐          ┌─────────────────────┐
  │ Step 4: EMBED   │          │ Step 6: EMBED       │
  │ embedder.py     │          │ same embedder       │
  │ HuggingFace     │          │                     │
  │ BGE-large 1024d │          │                     │
  └────────┬────────┘          └──────────┬──────────┘
           │                              │
           ▼                              ▼
  ┌─────────────────┐          ┌─────────────────────┐
  │ Step 5: UPLOAD  │          │ Step 6: UPLOAD      │
  │ uploader.py     │          │ RawTextUploader      │
  │ QdrantUploader  │          │                     │
  │ → school_intel  │          │ → school_intel_raw  │
  └────────┬────────┘          └──────────┬──────────┘
           │                              │
           └──────────┬───────────────────┘
                      ▼
           ┌─────────────────────┐
           │ Step 7: EXPORT      │  exporter.py
           │ Excel + JSON        │
           └─────────────────────┘
```

---

## File-by-File Breakdown

### `config.py` — Central configuration

All settings loaded from `.env` via `python-dotenv`. Key settings:

| Setting | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | API key for Gemini (extraction + optional embedding) |
| `LANGEXTRACT_MODEL` | `gemini-2.5-flash` | LLM used for entity extraction |
| `EMBEDDING_PROVIDER` | `gemini` | Choose: `gemini`, `openai`, `huggingface` |
| `EMBEDDING_MODEL` | `text-embedding-004` | Embedding model (currently using `BAAI/bge-large-en-v1.5`) |
| `QDRANT_URL` | — | Qdrant Cloud cluster URL |
| `QDRANT_COLLECTION` | `school_intel` | Entity collection name |
| `QDRANT_RAW_COLLECTION` | `school_intel_raw` | Raw text collection name |
| `USE_JS_RENDERING` | `false` | Use headless browser (StealthyFetcher) instead of plain HTTP |
| `RESPECT_ROBOTS_TXT` | `true` | Obey robots.txt rules (currently `false` in .env) |
| `MAX_PAGES_PER_SCHOOL` | `100` | Max pages to crawl (currently `250` in .env) |
| `PDF_BUDGET_RATIO` | `0.4` | Fraction of page budget for PDFs (currently `0.5`) |
| `RAW_CHUNK_SIZE` | `1000` | Characters per raw text chunk |
| `RAW_CHUNK_OVERLAP` | `200` | Overlap between raw text chunks |

---

### `spider.py` — Web crawler (~870 lines)

The largest and most complex file. Two crawl strategies:

**Strategy A: Sitemap-based (preferred)**
1. Tries `/sitemap.xml`, `/sitemap_index.xml`, `/sitemap/`, and `robots.txt → Sitemap:` directives
2. Parses all sitemap URLs, scores them for relevance using keyword matching
3. Splits budget: PDFs get `PDF_BUDGET_RATIO` of slots, HTML pages get the rest
4. Pages with relevance score < -5 are dropped
5. `SchoolSpider` crawls selected HTML pages and downloads selected PDFs
6. Also discovers and downloads any PDFs linked from crawled pages

**Strategy B: Fallback link-following**
- Used when no sitemap is found
- Starts at homepage, follows same-domain links
- Prioritizes links with high-value path keywords (`board`, `budget`, `vendor`, etc.)
- Strict page budget enforced

**Fetcher modes:**
- `USE_JS_RENDERING=false` → `Fetcher.get()` — plain HTTP, fast, no JavaScript
- `USE_JS_RENDERING=true` → `StealthyFetcher.fetch()` — headless Chromium browser, renders JS, anti-bot evasion (via camoufox/patchright)

**Key classes:**
- `RobotsChecker` — parses robots.txt, exposes `can_fetch()` and `crawl_delay`
- `SchoolSpider(Spider)` — sitemap-based crawler
- `FallbackSpider(Spider)` — link-following crawler
- `crawl_school()` — public entry point, picks strategy automatically

**URL scoring keywords:**
- HIGH (+10): board, trustee, governance, budget, vendor, contractor, procurement, etc.
- LOW (-8): news, blog, gallery, newsletter, login, cart, etc.
- PDF bonus (+20 base, +5 per keyword): minutes, report, budget, contract, tender

---

### `pdf_utils.py` — PDF text extraction

Extracts text page-by-page from downloaded PDFs.

**Extraction chain (tries in order):**
1. **pymupdf (fitz)** — fastest, primary
2. **pdfplumber** — fallback
3. **Tesseract OCR** — for scanned/image-based PDFs (requires Tesseract installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`)

Returns a `PDFDocument` with `pages: list[PDFPage]`, each containing the page number and extracted text.

---

### `extractor.py` — LLM entity extraction

Uses **LangExtract** library with **Gemini 2.5 Flash** to extract structured entities from text.

**Entity types extracted:**

| Type | Key Attributes |
|---|---|
| `vendor` | vendor_name, service_type, contract_value, expiry_date, status |
| `budget` | amount, currency, category, period, funding_source, status |
| `project` | project_name, description, value, timeline, status, vendor |
| `problem` | description, category, severity, date_mentioned, resolution |
| `board_member` | name, role, term_start, term_end |
| `contractor` | contractor_name, trade, project, contract_value, expiry_date |

**Extraction prompt rules:**
- Skip navigation, footers, cookie banners, "Powered by" credits
- Only extract vendors with evidence the school actually uses/pays them
- Require 2-4 sentence context in extraction_text
- Always fill service_type for vendors
- Don't hallucinate — use exact text from source

Extraction runs in parallel (4 workers via `ThreadPoolExecutor` in main.py).

---

### `chunker.py` — Entity → Qdrant chunk conversion

Converts `ExtractedEntity` objects into `Chunk` objects ready for embedding and upload.

**For each entity, builds:**
- `embed_text` — human-readable sentence (e.g. "Vendor: Google. Service: Google Workspace for Education. School: ACE Academy")
- `metadata` — all filterable fields + source tracing
- `chunk_id` — SHA256 hash of `school_name::type::source_url::source_page::text[:200]`

**Deduplication:**
Content-aware dedup by `(school_name, type, embed_text)`. When duplicates exist, keeps the chunk with the richest metadata (most non-empty fields). This prevents the same entity (e.g. "Amazon Smile") from appearing 28 times when extracted from 28 different pages.

---

### `raw_chunker.py` — Full-text sliding window chunker

A second pass that preserves the **raw page text** (not just extracted entities). This catches information that the LLM extraction might miss.

- Splits each page's text into overlapping windows (1000 chars, 200 overlap)
- Same treatment for PDF page texts
- Dedup by `(school_name, first 200 chars of text)`
- Stored in a separate Qdrant collection (`school_intel_raw`)

---

### `embedder.py` — Pluggable embedding

Three embedding providers:

| Provider | Model | Dimensions | Notes |
|---|---|---|---|
| `huggingface` (current) | `BAAI/bge-large-en-v1.5` | 1024 | Free, local, GPU-accelerated |
| `gemini` | `text-embedding-004` | 768 | Needs API key |
| `openai` | `text-embedding-3-small` | 1536 | Needs API key |

BGE models automatically get the `"Represent this sentence: "` prefix for documents. Batched embedding with progress logging.

---

### `uploader.py` — Qdrant Cloud upload

Two uploaders:

**`QdrantUploader`** — entity collection (`school_intel`)
- Creates collection if missing (cosine similarity)
- Creates payload indexes on `metadata.type` and `metadata.school_name` (KEYWORD index)
- Converts SHA256 chunk IDs to UUIDs for Qdrant
- Payload format: `{page_content: embed_text, metadata: {all_fields + source_tracing}}`
- Upsert-based — safe to re-run (deterministic IDs)
- Search via `client.query_points()` (qdrant-client v1.17+)

**`RawTextUploader(QdrantUploader)`** — raw text collection (`school_intel_raw`)
- Same infrastructure, different collection
- Payload: `{page_content: raw_text_window, metadata: {school_name, source_url, ...}}`

Source tracing metadata for every point:
```
source_url, source_type, source_domain, source_school_name,
source_page, source_filename, source_is_pdf, source_crawled_at,
source_chunk_text, source_label
```

---

### `exporter.py` — Excel + JSON export

- **Excel**: One sheet per entity type + summary sheet, auto-column-width
- **JSON**: Flat array of all chunks with IDs and metadata
- Files saved to `output/` with timestamp in filename

---

### `query.py` — CLI query tool

```bash
python query.py "vendor contracts expiring soon"
python query.py "roof problems" --type problem
python query.py --school "Auckland Academy" --type vendor
python query.py --list-schools
```

Embeds the query, searches entity collection, pretty-prints results by type.

---

### `query_notebook.ipynb` — Jupyter notebook

Interactive query interface with:
- `search(query)` — entity collection search
- `search_raw(query)` — raw text collection search
- `ask(question)` — RAG: merges results from both collections, sends to Gemini for answer
- `browse()` — scroll through records with filters
- Stats cells showing both collection sizes

---

### `push_json.py` — Re-upload existing JSON

Utility to re-embed and re-upload an existing JSON export to Qdrant. Useful for re-indexing after schema or embedding model changes.

```bash
python push_json.py output/Aceacademycharter_20260301_222117.json
```

---

### `main.py` — Pipeline orchestrator

```bash
python main.py --url https://www.aceacademycharter.org
python main.py --url https://... --name "Auckland Academy"
python main.py --csv schools.csv
python main.py --url https://... --no-qdrant
```

Runs all 7 steps in sequence. Saves checkpoints after extraction (JSON in `output/checkpoints/`). Supports batch processing from CSV (needs `url` column, optional `name`/`school_name` column). Reports per-stage timing at the end.

---

## Two Qdrant Collections

| Collection | Purpose | Point Payload |
|---|---|---|
| `school_intel` | Structured entities (vendor, budget, etc.) | `{page_content: "Vendor: Google...", metadata: {type, school_name, vendor_name, ...}}` |
| `school_intel_raw` | Raw page/PDF text windows | `{page_content: "The board approved...", metadata: {type: "raw_text", school_name, source_url, ...}}` |

The `ask()` function in the notebook queries **both** collections and merges results for RAG — entities give structured facts, raw text gives surrounding context.

---

## Data Flow Example

```
Input:  https://www.aceacademycharter.org

Crawl:  250 pages max, 125 PDF slots, 125 HTML slots
        → 45 pages scraped, 12 PDFs downloaded

Extract (Gemini 2.5 Flash):
        → 89 vendors, 5 budgets, 3 projects, 2 board_members, ...
        → After dedup: 41 unique vendors, ...

Chunk:  Each entity → embed_text + metadata + chunk_id
Raw:    45 pages → ~180 raw text chunks (1000 char windows)

Embed:  BAAI/bge-large-en-v1.5 → 1024-dim vectors

Upload: school_intel:     ~60 entity points
        school_intel_raw: ~180 raw text points

Export: output/Aceacademycharter_20260303_120000.xlsx
        output/Aceacademycharter_20260303_120000.json
```

---

## Known Issues & Missing Features

### Problems

1. **Shallow extraction data** — Many school websites simply don't publish detailed procurement data. Vendors are often just brand names ("Google", "Amazon") without contract values, expiry dates, or service descriptions. The extractor can only extract what's actually on the page.

2. **Duplicate extraction from multiple pages** — The same vendor (e.g. "Amazon Smile") can appear on dozens of pages. Content-aware dedup catches exact matches, but slightly different wording produces separate chunks. There's no fuzzy dedup or entity resolution.

3. **GPU not working** — The BAAI/bge-large-en-v1.5 embedding model falls back to CPU. The user's GTX 1650 has a CUDA driver that's too old for the current PyTorch. Needs driver update to CUDA 12+.

4. **robots.txt filter is commented out** — In `spider.py` around line 814, the robots.txt URL filter is commented out (`# all_entries = [e for e in all_entries if robots.can_fetch(e.url)]`), so even with `RESPECT_ROBOTS_TXT=true`, sitemap URLs are not actually filtered. Only dynamically discovered links and PDFs are filtered.

5. **JS rendering is slow** — When `USE_JS_RENDERING=true`, every page request launches a headless Chromium browser. For 250 pages this could take 30+ minutes vs 2-3 minutes with plain HTTP. There's no hybrid mode (e.g. try static first, fallback to JS for empty pages).

6. **SSL certificate errors with Fetcher** — `Fetcher.get()` (plain HTTP mode) sometimes fails with `curl: (60) SSL certificate problem` on certain environments/networks. StealthyFetcher doesn't have this issue since it uses a real browser.

7. **No incremental / delta crawl** — Re-running the pipeline re-crawls everything from scratch. There's no way to only fetch pages that changed since the last run. Qdrant upsert prevents duplicate points, but all the crawl + extraction + embedding work is repeated.

8. **PDF extraction quality** — OCR fallback hardcodes the Tesseract path to `C:\Program Files\Tesseract-OCR\tesseract.exe` (Windows-only). On Linux/Mac this path doesn't exist. Also, OCR quality on low-resolution scans is poor.

### Missing Features

1. **Multi-school dashboard / comparison** — No way to compare vendors, budgets, or contracts across multiple schools. The data is all in Qdrant but there's no aggregation or visualization layer.

2. **Scheduled / automated runs** — No cron job, task scheduler, or webhook trigger. Everything is manual CLI invocation.

3. **Authentication handling** — School websites behind login portals (parent portals, intranet) are completely inaccessible. The crawler has no login/session support.

4. **Table extraction** — Many school documents have structured data in HTML/PDF tables (budget breakdowns, vendor lists, contract tables). The current approach feeds raw text to the LLM, which often loses table structure. A dedicated table parser (e.g. camelot, tabula) would help.

5. **Entity resolution / linking** — "Google", "Google LLC", "Google Workspace", and "Google for Education" are all treated as separate vendors. There's no normalization or dedup at the entity level across schools.

6. **Contract monitoring / alerts** — No way to track contracts nearing expiry or flag pricing anomalies. The data is captured but there's no alerting layer.

7. **Web UI** — Everything is CLI or Jupyter notebook. No web-based search/browse interface for non-technical users.

8. **Rate limiting awareness** — The Gemini API has rate limits. With large crawls (250 pages × parallel extraction), the pipeline can hit Gemini's free-tier rate limits and fail with 429 errors. There's some retry logic but no adaptive rate limiting or token budgeting.

9. **Error recovery / partial re-run** — Checkpoints save after extraction, but there's no way to resume from a specific step. If embedding fails at chunk 150/200, you restart from scratch (or manually use `push_json.py` on the checkpoint).

10. **Test coverage** — There's a `tests/` directory but it was not populated during this session. No unit tests, integration tests, or validation tests exist.
