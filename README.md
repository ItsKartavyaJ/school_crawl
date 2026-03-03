# School Intelligence Crawler

Crawls school websites, extracts structured procurement intelligence (vendors, budgets, projects, board members, contractors), and stores it in a searchable vector database.

## Quick Start

```bash
# 1. Clone & setup
git clone <repo-url>
cd crawl
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install browser (only needed if USE_JS_RENDERING=true)
patchright install chromium
python -m camoufox fetch

# 4. Configure
copy .env.example .env
# Edit .env — add your GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY

# 5. Run
python main.py --url https://www.aceacademycharter.org
```

## Usage

```bash
# Single school
python main.py --url https://school-website.org

# With explicit name
python main.py --url https://school-website.org --name "My School"

# Batch from CSV (needs 'url' column, optional 'name' column)
python main.py --csv schools.csv

# Skip Qdrant upload, just export files
python main.py --url https://school-website.org --no-qdrant

# Query the database
python query.py "vendor contracts expiring soon"
python query.py "roof problems" --type problem
python query.py --list-schools
```

## Architecture

See [arch.md](arch.md) for the full architecture doc — file-by-file breakdown, data flow, and known issues.

**Pipeline:** Crawl → Extract (Gemini) → Chunk → Embed (BGE-large) → Upload (Qdrant) → Export (Excel/JSON)

**Two Qdrant collections:**
- `school_intel` — structured entities (vendors, budgets, projects, etc.)
- `school_intel_raw` — raw page text windows for deeper RAG context

## Key Config (.env)

| Setting | Description |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key (required) |
| `QDRANT_URL` | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `USE_JS_RENDERING` | `true` = headless browser, `false` = plain HTTP (default) |
| `MAX_PAGES_PER_SCHOOL` | Max pages to crawl per school (default: 250) |
| `EMBEDDING_MODEL` | Default: `BAAI/bge-large-en-v1.5` (1024 dims, free, local) |

## Project Structure

```
main.py            Pipeline orchestrator (7-step)
spider.py          Web crawler (sitemap + fallback link-following)
extractor.py       LLM entity extraction (LangExtract + Gemini)
chunker.py         Entity → vector-ready chunks + dedup
raw_chunker.py     Raw text sliding-window chunker
embedder.py        Pluggable embeddings (HuggingFace/Gemini/OpenAI)
uploader.py        Qdrant Cloud upload (entity + raw collections)
exporter.py        Excel + JSON export
pdf_utils.py       PDF text extraction (pymupdf → pdfplumber → OCR)
query.py           CLI search tool
push_json.py       Re-upload existing JSON exports
config.py          Central config from .env
query_notebook.ipynb  Jupyter notebook for interactive queries
arch.md            Architecture documentation
```
