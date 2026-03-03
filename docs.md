# School Intelligence Crawl - Docs

## What This Project Does
This pipeline crawls school websites, extracts structured entities, embeds them, and optionally uploads results to Qdrant for semantic search.

Main flow:
1. Discover sitemap and crawl selected pages/PDFs
2. Extract entities from web + PDF text (parallelized with ThreadPoolExecutor)
3. Convert entities into chunks (deduplicated via SHA256 hashes)
4. Generate embeddings (batched API calls with retry)
5. Upload to Qdrant (optional)
6. Export Excel + JSON files

---

## Requirements
- Python 3.12+ (recommended)
- Windows PowerShell (commands below assume PowerShell)
- API keys:
  - Gemini API key
  - Qdrant URL + API key (only needed if uploading to Qdrant)
- Optional for OCR of scanned PDFs:
  - Tesseract OCR installed and on PATH
  - `pytesseract` + `Pillow` Python packages (included in requirements.txt)

---

## Setup
From project root (`c:\Users\karta\Desktop\pintel\crawl`):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create your env file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set at least:
- `GEMINI_API_KEY`

If you want Qdrant upload, also set:
- `QDRANT_URL`
- `QDRANT_API_KEY`
- optional: `QDRANT_COLLECTION`

### Configuration Reference

All settings are in `.env` (see `.env.example` for defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Google AI Studio API key |
| `LANGEXTRACT_MODEL` | `gemini-2.5-flash` | Model for entity extraction |
| `EMBEDDING_PROVIDER` | `gemini` | `gemini`, `openai`, or `huggingface` |
| `EMBEDDING_MODEL` | `text-embedding-004` | Embedding model name |
| `MAX_PAGES_PER_SCHOOL` | `100` | Max pages to crawl per school |
| `PDF_BUDGET_RATIO` | `0.4` | Fraction of page budget for PDFs |
| `MIN_PAGE_TEXT_LENGTH` | `200` | Skip web pages with less text |
| `MIN_PDF_TEXT_LENGTH` | `50` | Skip PDF pages with less text |
| `RETRY_ATTEMPTS` | `3` | Max retries for API/network calls |
| `RAW_TEXT_MAX_LENGTH` | `1000` | Truncation limit for stored raw text |

---

## Running The Pipeline
### Single school (no Qdrant upload, local export only)
```powershell
python main.py --url "https://example-school.edu" --no-qdrant
```

### Single school (with Qdrant upload)
```powershell
python main.py --url "https://example-school.edu"
```

### Single school with explicit name
```powershell
python main.py --url "https://example-school.edu" --name "Example School"
```

### Multiple schools from CSV
CSV must include a `url` column. Optional columns: `name` or `school_name`.

```powershell
python main.py --csv schools.csv --no-qdrant
```

### Resume-friendly crawl directory
```powershell
python main.py --url "https://example-school.edu" --crawl-dir ".\output\checkpoints"
```

### Show CLI help
```powershell
python main.py --help
```

---

## Querying Stored Data (Qdrant)
After uploading, you can query via `query.py`.

```powershell
python query.py "vendor contracts expiring soon"
python query.py "roof problems" --type problem
python query.py --school "Example School" --type vendor
python query.py --list-schools
```

---

## Running Tests
```powershell
python -m pytest tests/ -v
```

---

## Output Files
Exports are written under the `output` directory.

- Excel files (multi-sheet: Vendors, Budgets, Projects, Problems, Board Members, Contractors)
- JSON exports with full metadata
- Downloaded PDFs in `output/pdfs/<domain>/`
- Extraction checkpoints in `output/checkpoints/` (re-run safe)

---

## Pipeline Timing
Each pipeline run logs per-stage timing:
```
  crawl          :   12.3s
  extraction     :   45.6s
  chunking       :    0.1s
  embedding      :    8.2s
  upload         :    3.1s
  export         :    0.5s
  TOTAL          :   69.8s
```

You should see:
- JSON export(s)
- Excel export(s)
- downloaded PDFs / crawl artifacts (depending on crawl path and source)

---

## Common Errors
### `QDRANT_URL is not set`
You ran without `--no-qdrant` and `.env` is missing Qdrant settings.

Fix:
- add `QDRANT_URL` + `QDRANT_API_KEY` to `.env`, or
- run with `--no-qdrant`

### Gemini key missing / auth failures
Set `GEMINI_API_KEY` correctly in `.env`.

### PowerShell script execution policy blocks activation
If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

## Quick Start (Copy/Paste)
```powershell
cd c:\Users\karta\Desktop\pintel\crawl
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env and set GEMINI_API_KEY
python main.py --url "https://example-school.edu" --no-qdrant
```
