"""
spider.py — Scrapling-based school website crawler

Flow:
  1. Discover sitemap  (tries /sitemap.xml, /sitemap_index.xml, robots.txt → Sitemap:)
  2. Parse ALL sitemap URLs + score each for relevance to school data
  3. Select the best pages to scrape (high-relevance first, strict MAX_PAGES_PER_SCHOOL cap)
  4. Route sitemap PDFs directly to _download_pdf, pages to parse()
  5. Crawl only selected pages + download all PDFs found on them

Bug fixes applied:
  [1] parse_sitemap: bad <priority> values no longer abort the whole file
  [2] SchoolSpider: sitemap PDFs routed directly to _download_pdf, not parse()
  [3] FallbackSpider: strict page budget enforced before enqueueing links
  [4] PDF filenames: URL-hash suffix added to prevent basename collisions
  [5] select_pages: PDFs capped at PDF_BUDGET, remaining slots for pages
"""

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from typing import Optional

from loguru import logger
from scrapling.fetchers import Fetcher, StealthyFetcher
from scrapling.spiders import Spider, Request, Response
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import (
    MAX_PAGES_PER_SCHOOL,
    CONCURRENT_REQUESTS,
    DOWNLOAD_DELAY,
    PRIORITY_PATHS,
    OUTPUT_DIR,
    PDF_BUDGET_RATIO,
    DOWNLOAD_ALL_PDFS,
    ALLOWED_EXTERNAL_PDF_HOSTS,
    RESPECT_ROBOTS_TXT,
    USE_JS_RENDERING,
    RETRY_ATTEMPTS,
    RETRY_WAIT_MIN,
    RETRY_WAIT_MAX,
)


def _bare_domain(netloc: str) -> str:
    """Strip optional 'www.' prefix so isd622.org == www.isd622.org."""
    return netloc.lower().removeprefix("www.")


def _same_domain(a: str, b: str) -> bool:
    """Check two netlocs belong to the same domain (ignoring www.)."""
    return _bare_domain(a) == _bare_domain(b)


def _host_allowed_for_pdf(host: str, school_domain: str) -> bool:
    """
    Allow PDFs from same-domain host plus configured external host allowlist.
    """
    host = (host or "").lower()
    if not host:
        return False
    if _same_domain(host, school_domain):
        return True
    for allowed in ALLOWED_EXTERNAL_PDF_HOSTS:
        if host == allowed or host.endswith(f".{allowed}"):
            return True
    return False


def _is_allowed_pdf_url(pdf_url: str, school_domain: str) -> bool:
    """Basic URL sanity + host allowlist check for PDF links."""
    try:
        p = urlparse(pdf_url)
    except Exception:
        return False
    if p.scheme not in ("http", "https"):
        return False
    if not p.netloc:
        return False
    return _host_allowed_for_pdf(p.netloc, school_domain)


# ── Dynamic JS rendering detection ──────────────────────────────────────────────

# Indicators that a page relies on client-side JS to render meaningful content.
_JS_FRAMEWORK_MARKERS = [
    "__NEXT_DATA__",       # Next.js
    "__NUXT__",            # Nuxt / Vue SSR
    "data-reactroot",      # React
    'id="__next"',         # Next.js root
    'id="root"></div>',    # CRA / Vite React
    'id="app"></div>',     # Vue / generic SPA
    "ng-app",              # AngularJS
    "ng-version",          # Angular 2+
    "window.__remixContext",  # Remix
    "_ssgManifest",        # Next.js SSG
    "data-turbo",          # Hotwire Turbo
]

# Minimum visible-text length (after stripping tags) to consider a
# page "content-rich enough" without JS.
_JS_DETECT_MIN_TEXT = 300


def _get_response_text(resp) -> str:
    """Extract HTML text from a Scrapling response, regardless of attribute name."""
    for attr in ("content", "text", "html_content", "body"):
        val = getattr(resp, attr, None)
        if val:
            return str(val) if isinstance(val, (str, bytes)) else val.decode("utf-8", errors="replace") if isinstance(val, (bytes, bytearray)) else str(val)
    return ""


def _detect_needs_js(url: str) -> bool:
    """
    Probe a URL with plain HTTP and decide whether JS rendering is needed.

    Returns True when ANY of these hold:
      1. The site returns 403/406/429 (bot-blocked) — needs a real browser
      2. Visible text is very thin AND the HTML contains a JS-framework marker

    Returns False when:
      - The static page has enough content and no framework markers
      - The probe fails entirely (default to static to avoid unnecessary browser)
    """
    try:
        resp = Fetcher.get(url, stealthy_headers=True, timeout=12)
        if not resp:
            return False          # can't tell — stay with static

        status = getattr(resp, "status", 0) or 0

        # ── Bot-blocking status codes → almost certainly needs a real browser ──
        if status in (403, 406, 429):
            logger.info(
                f"JS auto-detect: ENABLED for {urlparse(url).netloc} "
                f"(static fetch returned HTTP {status} — likely bot-blocked)"
            )
            return True

        if status != 200:
            return False          # other errors — can't tell, stay static

        raw_html = _get_response_text(resp)

        # Strip tags to get rough visible text
        visible = re.sub(r"<[^>]+>", " ", raw_html)
        visible = re.sub(r"\s+", " ", visible).strip()

        has_marker = any(m in raw_html for m in _JS_FRAMEWORK_MARKERS)
        is_thin    = len(visible) < _JS_DETECT_MIN_TEXT

        if is_thin and has_marker:
            logger.info(
                f"JS auto-detect: ENABLED for {urlparse(url).netloc} "
                f"(visible text: {len(visible)} chars, framework marker found)"
            )
            return True

        logger.info(
            f"JS auto-detect: disabled for {urlparse(url).netloc} "
            f"(visible text: {len(visible)} chars, marker={has_marker})"
        )
        return False

    except Exception as exc:
        logger.debug(f"JS auto-detect probe failed ({exc}) — defaulting to static")
        return False


# ── Retry-wrapped Fetcher ───────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True,
)
def _fetch_with_retry(url: str, timeout: int = 15, use_js: bool = False):
    """Fetch a URL with automatic retry on transient network errors.

    When use_js is True, uses a **hybrid strategy**:
      1. Try plain HTTP Fetcher first (fast)
      2. If the response body is very thin (<500 chars), fall back to
         StealthyFetcher (headless Chromium) to render JavaScript.
    This avoids launching a browser for pages that work fine statically,
    cutting JS rendering invocations by ~80% on typical school sites.
    """
    if use_js:
        # --- Try static first ---
        try:
            resp = Fetcher.get(url, stealthy_headers=True, timeout=timeout)
            body_text = ""
            if resp and (getattr(resp, "status", 0) or 0) == 200:
                body_text = _get_response_text(resp)
            if len(body_text.strip()) >= 500:
                return resp          # static page has enough content
            logger.debug(f"Thin static response ({len(body_text)} chars), using JS: {url}")
        except Exception:
            logger.debug(f"Static fetch failed, falling back to JS: {url}")

        # --- Fall back to headless browser ---
        resp = StealthyFetcher.fetch(
            url,
            headless=True,
            network_idle=True,
            timeout=timeout * 1000,   # StealthyFetcher uses milliseconds
            disable_resources=True,   # skip fonts/images/media for speed
        )
        # Compatibility: StealthyFetcher uses .html_content, Fetcher uses .content
        if not hasattr(resp, "content"):
            resp.content = resp.html_content
        return resp
    return Fetcher.get(url, stealthy_headers=True, timeout=timeout)


# ── Robots.txt compliance ─────────────────────────────────────────────────────

class RobotsChecker:
    """
    Fetches and parses robots.txt for a domain, then exposes:
      - can_fetch(url)   → bool
      - crawl_delay      → float (seconds, default DOWNLOAD_DELAY)

    Uses stdlib urllib.robotparser which handles User-agent, Allow,
    Disallow, and Crawl-delay directives correctly.
    """

    USER_AGENT = "*"  # our bot doesn't declare a specific UA

    def __init__(self, base_url: str, use_js: bool = False):
        parsed = urlparse(base_url)
        self._robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        self._parser = RobotFileParser()
        self._loaded = False
        self._crawl_delay: Optional[float] = None
        self._load(use_js=use_js)

    def _load(self, use_js: bool = False):
        """Fetch and parse robots.txt. On failure, allow everything."""
        try:
            resp = _fetch_with_retry(self._robots_url, timeout=10, use_js=use_js)
            if resp and resp.status == 200:
                content = str(resp.content or "")
                lines = content.splitlines()
                self._parser.parse(lines)
                # Extract Crawl-delay manually (RobotFileParser supports it
                # via .crawl_delay() in Python 3.6+)
                try:
                    delay = self._parser.crawl_delay(self.USER_AGENT)
                    if delay is not None:
                        self._crawl_delay = float(delay)
                except Exception:
                    pass
                self._loaded = True
                n_disallow = sum(1 for ln in lines
                                 if ln.strip().lower().startswith("disallow:"))
                logger.info(
                    f"robots.txt loaded: {n_disallow} Disallow rules, "
                    f"Crawl-delay={self._crawl_delay}"
                )
            else:
                logger.info("No robots.txt found — all URLs allowed")
        except Exception as e:
            logger.warning(f"Could not fetch robots.txt: {e} — all URLs allowed")

    def can_fetch(self, url: str) -> bool:
        """Return True if the URL is allowed by robots.txt."""
        if not self._loaded:
            return True
        return self._parser.can_fetch(self.USER_AGENT, url)

    @property
    def crawl_delay(self) -> float:
        """Crawl delay in seconds. Falls back to DOWNLOAD_DELAY from config."""
        if self._crawl_delay is not None:
            return self._crawl_delay
        return DOWNLOAD_DELAY

    def filter_urls(self, urls: list[str]) -> tuple[list[str], int]:
        """Filter a list of URLs, returning (allowed, blocked_count)."""
        allowed = [u for u in urls if self.can_fetch(u)]
        blocked = len(urls) - len(allowed)
        return allowed, blocked


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SitemapURL:
    url: str
    lastmod: str = ""
    priority: float = 0.5
    relevance_score: int = 0
    is_pdf: bool = False


@dataclass
class PageResult:
    url: str
    text: str
    html: str
    title: str = ""
    source_type: str = "website"


@dataclass
class PDFResult:
    url: str
    local_path: str
    filename: str
    source_type: str = "pdf_document"


@dataclass
class SchoolCrawlResult:
    school_name: str
    domain: str
    sitemap_urls_found: int = 0
    sitemap_urls_selected: int = 0
    pages: list[PageResult] = field(default_factory=list)
    pdfs:  list[PDFResult]  = field(default_factory=list)

    @property
    def total_sources(self):
        return len(self.pages) + len(self.pdfs)


# ── Keyword scoring sets ──────────────────────────────────────────────────────

HIGH_VALUE_KEYWORDS = [
    "board", "trustee", "governance", "minutes", "meeting",
    "budget", "finance", "financial", "annual-report", "annual_report",
    "vendor", "supplier", "contractor", "procurement", "tender", "contract",
    "project", "capital", "infrastructure",
    "staff", "team", "about", "leadership", "principal",
    "policy", "policies", "bylaws",
]

LOW_VALUE_KEYWORDS = [
    "news", "blog", "post", "article", "event", "calendar",
    "gallery", "photo", "image", "video", "media",
    "enrolment", "enroll", "uniform", "sport", "camp",
    "newsletter", "archive", "tag", "category", "author",
    "login", "logout", "register", "account", "cart", "shop",
    "sitemap", "feed", "rss", "print", "comment",
]

PDF_BONUS_KEYWORDS = ["minutes", "report", "budget", "contract", "tender", "procurement"]


# ── PDF filename helper ───────────────────────────────────────────────────────

def _safe_pdf_filename(url: str) -> str:
    """
    FIX [4]: Generate a collision-safe filename for a PDF URL.

    Appends a short hash of the full URL to the basename so that two different
    URLs with the same filename (e.g. /2023/report.pdf and /2024/report.pdf)
    do NOT silently overwrite each other.

    Examples:
        https://school.nz/docs/report.pdf         → report_a3f2c1.pdf
        https://school.nz/old/report.pdf          → report_9d1e4b.pdf
        https://school.nz/board/minutes-2024.pdf  → minutes-2024_7c8a21.pdf
    """
    basename = url.split("/")[-1].split("?")[0] or "document"
    if not basename.lower().endswith(".pdf"):
        basename += ".pdf"
    stem = Path(basename).stem
    suffix = hashlib.sha256(url.encode()).hexdigest()[:6]
    return f"{stem}_{suffix}.pdf"


# ── Sitemap discovery ─────────────────────────────────────────────────────────

def discover_sitemaps(base_url: str, use_js: bool = False) -> list[str]:
    """
    Find all sitemap URLs for a website.

    Tries in order:
      1. /sitemap.xml
      2. /sitemap_index.xml
      3. /sitemap/
      4. robots.txt → 'Sitemap:' directives
    """
    parsed   = urlparse(base_url)
    root_url = f"{parsed.scheme}://{parsed.netloc}"

    candidates = [
        f"{root_url}/sitemap.xml",
        f"{root_url}/sitemap_index.xml",
        f"{root_url}/sitemap/",
        f"{root_url}/sitemap",
    ]

    found: list[str] = []

    for url in candidates:
        try:
            page = _fetch_with_retry(url, timeout=10, use_js=use_js)
            if page and page.status == 200:
                content = str(page.content or "")
                if "<urlset" in content or "<sitemapindex" in content:
                    logger.success(f"Sitemap found: {url}")
                    found.append(url)
                    break
        except Exception:
            continue

    # Also check robots.txt for Sitemap: directives
    try:
        robots_url = f"{root_url}/robots.txt"
        page = _fetch_with_retry(robots_url, timeout=10, use_js=use_js)
        if page and page.status == 200:
            for line in str(page.content or "").splitlines():
                line = line.strip()
                if line.lower().startswith("sitemap:"):
                    sm_url = line.split(":", 1)[1].strip()
                    if sm_url not in found:
                        logger.info(f"Sitemap from robots.txt: {sm_url}")
                        found.append(sm_url)
    except Exception:
        pass

    if not found:
        logger.warning(f"No sitemap found for {base_url} — will use link-following fallback")

    return found


def _parse_priority(raw: Optional[str]) -> float:
    """
    FIX [1]: Safely parse a sitemap <priority> value.
    Returns 0.5 (default) on any parse failure instead of throwing.
    """
    if not raw:
        return 0.5
    try:
        value = float(raw.strip())
        # Clamp to valid range
        return max(0.0, min(1.0, value))
    except (ValueError, TypeError):
        logger.debug(f"Non-numeric <priority> value ignored: {raw!r}")
        return 0.5


def parse_sitemap(sitemap_url: str, domain: str, depth: int = 0, use_js: bool = False) -> list[SitemapURL]:
    """
    Parse a sitemap XML and return all page URLs.
    Handles sitemap index files recursively (up to depth 2).

    FIX [1]: Bad <priority> values are handled per-entry and never abort the file.
             Malformed individual <url> entries are skipped, not the whole sitemap.
    """
    if depth > 2:
        return []

    urls: list[SitemapURL] = []

    try:
        page = _fetch_with_retry(sitemap_url, timeout=15, use_js=use_js)
        if not page or page.status != 200:
            logger.warning(f"Could not fetch sitemap: {sitemap_url} (status {getattr(page, 'status', 'N/A')})")
            return []

        content = str(page.content or "")

        # Strip XML namespaces so ElementTree can parse cleanly
        content = re.sub(r'\sxmlns[^=]*="[^"]*"', '', content)
        content = re.sub(r'<(\/?)[\w-]+:', r'<\1', content)

        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.warning(f"XML parse error in {sitemap_url}: {e} — skipping")
            return []

        # ── Sitemap index → recurse into child sitemaps ────────────────────
        if "sitemapindex" in root.tag.lower():
            logger.info(f"Sitemap index at {sitemap_url} — fetching child sitemaps...")
            for sm in root.findall(".//sitemap"):
                loc = (sm.findtext("loc") or "").strip()
                if loc and urlparse(loc).netloc == domain:
                    urls.extend(parse_sitemap(loc, domain, depth + 1, use_js=use_js))

        # ── Regular sitemap → extract page URLs ───────────────────────────
        else:
            for url_elem in root.findall(".//url"):
                try:
                    loc     = (url_elem.findtext("loc")     or "").strip()
                    lastmod = (url_elem.findtext("lastmod") or "").strip()
                    # FIX [1]: use safe parser, never throws
                    priority = _parse_priority(url_elem.findtext("priority"))

                    if not loc:
                        continue
                    if urlparse(loc).netloc != domain:
                        continue

                    urls.append(SitemapURL(
                        url=loc,
                        lastmod=lastmod,
                        priority=priority,
                        is_pdf=".pdf" in loc.lower(),
                    ))

                except Exception as e:
                    # FIX [1]: skip this entry, not the whole sitemap
                    logger.debug(f"Skipping malformed <url> entry: {e}")
                    continue

        logger.info(f"  → {len(urls)} URLs from {sitemap_url}")

    except Exception as e:
        logger.error(f"Sitemap fetch/parse failed {sitemap_url}: {e}")

    return urls


# ── Relevance scoring ─────────────────────────────────────────────────────────

def score_url(entry: SitemapURL) -> SitemapURL:
    """Compute a relevance score. Higher = more likely to contain school data."""
    url_lower = entry.url.lower()
    score = 0

    for kw in HIGH_VALUE_KEYWORDS:
        if kw in url_lower:
            score += 10

    if entry.is_pdf:
        score += 20
        for kw in PDF_BONUS_KEYWORDS:
            if kw in url_lower:
                score += 5

    for kw in LOW_VALUE_KEYWORDS:
        if kw in url_lower:
            score -= 8

    depth = len([p for p in urlparse(entry.url).path.split("/") if p])
    if depth <= 2:
        score += 5
    elif depth >= 5:
        score -= 3

    # Sitemap-declared priority scaled to 0–5
    score += int(entry.priority * 5)

    entry.relevance_score = score
    return entry


def select_pages(
    sitemap_entries: list[SitemapURL],
    max_pages: int,
) -> tuple[list[SitemapURL], list[SitemapURL]]:
    """
    Select the most relevant sitemap URLs to crawl, strictly within max_pages.

    FIX [5]: PDFs are capped at PDF_BUDGET_RATIO * max_pages so they can't
             consume the entire crawl budget and crowd out HTML pages.

    Returns:
        (selected_pages, selected_pdfs) — two separate lists for routing.
    """
    scored = [score_url(e) for e in sitemap_entries]

    pdf_entries  = sorted([e for e in scored if e.is_pdf],
                          key=lambda e: e.relevance_score, reverse=True)
    page_entries = sorted([e for e in scored if not e.is_pdf],
                          key=lambda e: e.relevance_score, reverse=True)

    # Strict budget split unless configured to keep all PDFs.
    if DOWNLOAD_ALL_PDFS:
        selected_pdfs = pdf_entries
        page_budget = max_pages
        selected_pages = [p for p in page_entries if p.relevance_score >= -5][:page_budget]
        pdf_budget_label = "ALL"
    else:
        pdf_budget = int(max_pages * PDF_BUDGET_RATIO)
        page_budget = max_pages - pdf_budget
        selected_pdfs = pdf_entries[:pdf_budget]
        selected_pages = [p for p in page_entries if p.relevance_score >= -5][:page_budget]
        pdf_budget_label = str(pdf_budget)

    # Log distribution
    high   = sum(1 for e in page_entries if e.relevance_score > 10)
    medium = sum(1 for e in page_entries if 0 <= e.relevance_score <= 10)
    low    = sum(1 for e in page_entries if e.relevance_score < 0)
    logger.info(
        f"Sitemap: {high} high-value pages | {medium} medium | {low} low-value | "
        f"{len(pdf_entries)} PDFs total"
    )
    logger.info(
        f"Selected: {len(selected_pages)} pages (budget {page_budget}) + "
        f"{len(selected_pdfs)} PDFs (budget {pdf_budget_label}) "
        f"= {len(selected_pages) + len(selected_pdfs)} total"
    )

    top = sorted(selected_pages + selected_pdfs,
                 key=lambda e: e.relevance_score, reverse=True)[:15]
    for e in top:
        tag = "PDF" if e.is_pdf else "   "
        logger.debug(f"  [{e.relevance_score:+3d}] [{tag}] {e.url}")

    return selected_pages, selected_pdfs


# ── Helpers ───────────────────────────────────────────────────────────────────

def classify_pdf(url: str, filename: str) -> str:
    text = (url + " " + filename).lower()
    if any(k in text for k in ["minute", "meeting", "board", "trustee"]):
        return "board_meeting"
    if any(k in text for k in ["budget", "finance", "financial", "annual-report", "annual_report"]):
        return "annual_report"
    if any(k in text for k in ["tender", "contract", "procurement", "rfp", "vendor"]):
        return "tender_doc"
    if any(k in text for k in ["project", "capital", "infrastructure", "build"]):
        return "project_doc"
    return "pdf_document"


def classify_page(url: str, title: str, text: str) -> str:
    """Shared page classification used by both SchoolSpider and FallbackSpider."""
    combined = (url + " " + title + " " + text[:200]).lower()
    if any(k in combined for k in ["board meeting", "minutes", "trustees"]):
        return "board_meeting"
    if any(k in combined for k in ["annual report", "financial statement"]):
        return "annual_report"
    if any(k in combined for k in ["budget", "appropriation", "financial summary"]):
        return "budget_page"
    if any(k in combined for k in ["vendor", "supplier", "contractor", "tender", "procurement"]):
        return "vendor_page"
    if any(k in combined for k in ["project", "capital works", "construction"]):
        return "project_page"
    return "website"


def url_priority(url: str) -> int:
    path = urlparse(url).path.lower()
    for i, kw in enumerate(PRIORITY_PATHS):
        if kw in path:
            return len(PRIORITY_PATHS) - i
    return 0


# ── Sitemap-based spider ──────────────────────────────────────────────────────

class SchoolSpider(Spider):
    """
    Crawls pre-selected HTML pages and directly downloads pre-selected PDFs.

    FIX [2]: Sitemap PDFs are passed in separately and scheduled directly
             to _download_pdf — they never go through parse() as HTML pages.
    """

    name = "school_spider"
    concurrent_requests = CONCURRENT_REQUESTS

    def __init__(
        self,
        start_url: str,
        school_name: str,
        selected_pages: list[str],    # HTML pages from sitemap
        selected_pdfs: list[str],     # PDF URLs from sitemap → direct download
        crawl_dir: Optional[str] = None,
        robots: Optional[RobotsChecker] = None,
        use_js: bool = False,
    ):
        self.start_url   = start_url
        self.school_name = school_name
        self.domain      = urlparse(start_url).netloc
        self._robots     = robots
        self._use_js     = use_js

        self._pages:     list[PageResult] = []
        self._pdfs:      list[PDFResult]  = []
        # Pre-seed seen set with everything already scheduled
        self._seen_urls: set[str] = set(selected_pages) | set(selected_pdfs)
        self._page_count = 0

        self._pdf_dir = OUTPUT_DIR / "pdfs" / self.domain.replace(".", "_")
        self._pdf_dir.mkdir(parents=True, exist_ok=True)

        # FIX [2]: HTML pages use parse(), PDF URLs use _download_pdf directly
        self.start_urls = selected_pages or [start_url]
        self._sitemap_pdf_urls = selected_pdfs   # scheduled in run()

        super().__init__(crawldir=crawl_dir)

    async def parse(self, response: Response):
        if self._page_count >= MAX_PAGES_PER_SCHOOL:
            return

        url = str(response.url)
        self._page_count += 1
        logger.info(f"[{self._page_count}] Scraping: {url}")

        title       = response.css("title::text").get() or ""
        text        = response.get_all_text(separator="\n", strip=True)
        html        = str(response.html_content) if hasattr(response, "html_content") else ""
        source_type = classify_page(url, title, text)

        self._pages.append(PageResult(
            url=url, text=text, html=html,
            title=title, source_type=source_type,
        ))

        # Collect any PDF links on this page not already scheduled
        pdf_hrefs = set(
            response.css("a[href$='.pdf']::attr(href)").getall()
            + response.css("a[href*='.pdf?']::attr(href)").getall()
            + response.xpath("//a[contains(@href, '.pdf')]/@href").getall()
        )
        for href in pdf_hrefs:
            pdf_url = urljoin(url, href)
            if not _is_allowed_pdf_url(pdf_url, self.domain):
                logger.debug(f"Skipped external/non-http PDF URL: {pdf_url}")
                continue
            if pdf_url not in self._seen_urls:
                # Respect robots.txt for dynamically discovered PDFs
                if self._robots and not self._robots.can_fetch(pdf_url):
                    logger.debug(f"robots.txt blocked PDF: {pdf_url}")
                    continue
                self._seen_urls.add(pdf_url)
                yield Request(pdf_url, callback=self._download_pdf)

    async def _download_pdf(self, response: Response):
        pdf_url  = str(response.url)
        # FIX [4]: collision-safe filename
        filename   = _safe_pdf_filename(pdf_url)
        local_path = self._pdf_dir / filename

        try:
            content = response.body if hasattr(response, "body") else b""
            if not content:
                raw     = _fetch_with_retry(pdf_url, timeout=20, use_js=self._use_js)
                content = raw.body if hasattr(raw, "body") else b""

            if content:
                local_path.write_bytes(content)
                logger.success(f"PDF saved: {filename}")
                self._pdfs.append(PDFResult(
                    url=pdf_url,
                    local_path=str(local_path),
                    filename=filename,
                    source_type=classify_pdf(pdf_url, filename),
                ))
            else:
                logger.warning(f"Empty PDF body: {pdf_url}")
        except Exception as e:
            logger.error(f"PDF download failed {pdf_url}: {e}")
        # Scrapling engine iterates callbacks with `async for`;
        # a yield makes this an async generator instead of a plain coroutine.
        return
        yield  # noqa: unreachable – required to satisfy async-generator protocol

    def run(self) -> SchoolCrawlResult:
        logger.info(
            f"Crawling {self.school_name}: "
            f"{len(self.start_urls)} HTML pages + "
            f"{len(self._sitemap_pdf_urls)} PDFs"
        )
        # FIX [2]: inject sitemap PDFs as explicit PDF requests before crawl starts
        # Scrapling allows adding extra start requests by appending to start_requests
        # We do this by adding them into start_urls with a special callback marker.
        # Since Scrapling Spider doesn't natively support per-URL callbacks in
        # start_urls, we download sitemap PDFs directly via Fetcher before crawl.
        for pdf_url in self._sitemap_pdf_urls:
            try:
                if not _is_allowed_pdf_url(pdf_url, self.domain):
                    logger.debug(f"Skipped sitemap PDF from disallowed host: {pdf_url}")
                    continue
                # Respect Crawl-delay between sitemap PDF downloads
                if self._robots:
                    time.sleep(self._robots.crawl_delay)
                raw = _fetch_with_retry(pdf_url, timeout=20, use_js=self._use_js)
                content = raw.body if hasattr(raw, "body") else b""
                if content:
                    filename   = _safe_pdf_filename(pdf_url)
                    local_path = self._pdf_dir / filename
                    local_path.write_bytes(content)
                    logger.success(f"Sitemap PDF saved: {filename}")
                    self._pdfs.append(PDFResult(
                        url=pdf_url,
                        local_path=str(local_path),
                        filename=filename,
                        source_type=classify_pdf(pdf_url, filename),
                    ))
                else:
                    logger.warning(f"Empty sitemap PDF: {pdf_url}")
            except Exception as e:
                logger.error(f"Sitemap PDF download failed {pdf_url}: {e}")

        self.start()
        logger.success(
            f"Done: {len(self._pages)} pages scraped, {len(self._pdfs)} PDFs downloaded"
        )
        return SchoolCrawlResult(
            school_name=self.school_name,
            domain=self.domain,
            pages=self._pages,
            pdfs=self._pdfs,
        )


# ── Fallback spider (no sitemap) ──────────────────────────────────────────────

class FallbackSpider(Spider):
    """
    Used when no sitemap is found.
    Follows links from homepage, prioritising high-value paths.

    FIX [3]: Strict page budget enforced BEFORE enqueueing links.
             We track total queued count and stop adding once budget is reached.
    """

    name = "fallback_spider"
    concurrent_requests = CONCURRENT_REQUESTS

    def __init__(
        self,
        start_url: str,
        school_name: str,
        crawl_dir: Optional[str] = None,
        robots: Optional[RobotsChecker] = None,
    ):
        self.start_url   = start_url
        self.school_name = school_name
        self.domain      = urlparse(start_url).netloc
        self._robots     = robots

        self._pages:        list[PageResult] = []
        self._pdfs:         list[PDFResult]  = []
        self._seen_urls:    set[str]         = {start_url}
        self._page_count:   int              = 0
        # FIX [3]: track total queued (fetched + pending) to enforce strict cap
        self._queued_count: int              = 1   # homepage already queued

        self._pdf_dir = OUTPUT_DIR / "pdfs" / self.domain.replace(".", "_")
        self._pdf_dir.mkdir(parents=True, exist_ok=True)

        self.start_urls = [start_url]
        super().__init__(crawldir=crawl_dir)

    async def parse(self, response: Response):
        url = str(response.url)
        self._page_count += 1
        logger.info(f"[{self._page_count}/{MAX_PAGES_PER_SCHOOL}] Crawling: {url}")

        title       = response.css("title::text").get() or ""
        text        = response.get_all_text(separator="\n", strip=True)
        html        = str(response.html_content) if hasattr(response, "html_content") else ""

        source_type = classify_page(url, title, text)

        self._pages.append(PageResult(url=url, text=text, html=html,
                                      title=title, source_type=source_type))

        # PDFs (not counted against page budget)
        for href in set(
            response.css("a[href$='.pdf']::attr(href)").getall()
            + response.xpath("//a[contains(@href,'.pdf')]/@href").getall()
        ):
            pdf_url = urljoin(url, href)
            if not _is_allowed_pdf_url(pdf_url, self.domain):
                logger.debug(f"Skipped external/non-http PDF URL: {pdf_url}")
                continue
            if pdf_url not in self._seen_urls:
                if self._robots and not self._robots.can_fetch(pdf_url):
                    logger.debug(f"robots.txt blocked PDF: {pdf_url}")
                    continue
                self._seen_urls.add(pdf_url)
                yield Request(pdf_url, callback=self._download_pdf)

        # FIX [3]: only enqueue more pages if queued count is still under budget
        if self._queued_count >= MAX_PAGES_PER_SCHOOL:
            return

        links = []
        for href in response.css("a::attr(href)").getall():
            full = urljoin(url, href)
            p    = urlparse(full)
            if (p.scheme.startswith("http")
                    and _same_domain(p.netloc, self.domain)
                    and full not in self._seen_urls
                    and not p.path.endswith((".jpg", ".png", ".gif", ".css", ".js", ".xml"))):
                # Respect robots.txt for discovered links
                if self._robots and not self._robots.can_fetch(full):
                    logger.debug(f"robots.txt blocked link: {full}")
                    continue
                links.append(full)

        # Sort high-value first, then take only what fits in remaining budget
        links.sort(key=url_priority, reverse=True)
        remaining = MAX_PAGES_PER_SCHOOL - self._queued_count

        for link in links[:remaining]:
            self._seen_urls.add(link)
            self._queued_count += 1
            yield response.follow(link, callback=self.parse)

    async def _download_pdf(self, response: Response):
        pdf_url    = str(response.url)
        # FIX [4]: collision-safe filename
        filename   = _safe_pdf_filename(pdf_url)
        local_path = self._pdf_dir / filename
        try:
            content = response.body if hasattr(response, "body") else b""
            if content:
                local_path.write_bytes(content)
                self._pdfs.append(PDFResult(
                    url=pdf_url, local_path=str(local_path),
                    filename=filename, source_type=classify_pdf(pdf_url, filename),
                ))
        except Exception as e:
            logger.error(f"PDF download failed {pdf_url}: {e}")
        # Scrapling engine iterates callbacks with `async for`;
        # a yield makes this an async generator instead of a plain coroutine.
        return
        yield  # noqa: unreachable – required to satisfy async-generator protocol

    def run(self) -> SchoolCrawlResult:
        self.start()
        return SchoolCrawlResult(
            school_name=self.school_name,
            domain=self.domain,
            pages=self._pages,
            pdfs=self._pdfs,
        )


# ── Public entry point ────────────────────────────────────────────────────────

def crawl_school(
    url: str,
    school_name: str = "",
    crawl_dir: Optional[str] = None,
) -> SchoolCrawlResult:
    """
    Sitemap-first crawl of a school website.

    1. Load robots.txt (respect Disallow / Allow / Crawl-delay)
    2. Discover sitemap(s)
    3. Parse all URLs from sitemap
    4. Filter through robots.txt
    5. Score + select best pages (strict budget split: pages vs PDFs)
    6. Route HTML pages → parse(), PDF URLs → _download_pdf() directly
    7. Return pages + PDFs

    Falls back to strict-budget link-following if no sitemap found.
    """
    if not school_name:
        school_name = urlparse(url).netloc.replace("www.", "").split(".")[0].title()

    domain = urlparse(url).netloc

    # ── Resolve JS rendering mode ──────────────────────────────────────────────
    use_js = False
    if USE_JS_RENDERING == "auto":
        use_js = _detect_needs_js(url)
        logger.info(f"JS rendering: {'ON' if use_js else 'OFF'} (auto-detected)")
    elif isinstance(USE_JS_RENDERING, bool):
        use_js = USE_JS_RENDERING
        logger.info(f"JS rendering: {'ON' if use_js else 'OFF'} (static config)")

    # ── Load robots.txt (skip if RESPECT_ROBOTS_TXT is False) ──────────────────
    if RESPECT_ROBOTS_TXT:
        robots = RobotsChecker(url, use_js=use_js)
        logger.info(f"Effective crawl delay: {robots.crawl_delay}s")
    else:
        robots = None
        logger.info("robots.txt compliance DISABLED — all URLs allowed")

    # ── Discover sitemaps ──────────────────────────────────────────────────────
    logger.info(f"Looking for sitemap: {domain}")
    sitemap_urls = discover_sitemaps(url, use_js=use_js)

    if sitemap_urls:
        # ── Parse all sitemap URLs ─────────────────────────────────────────────
        all_entries: list[SitemapURL] = []
        for sm_url in sitemap_urls:
            entries = parse_sitemap(sm_url, domain, use_js=use_js)
            all_entries.extend(entries)

        logger.info(f"Total sitemap URLs: {len(all_entries)}")

        # ── Filter through robots.txt ──────────────────────────────────────────
        blocked = 0
        if robots:
            pre_count = len(all_entries)
            all_entries = [e for e in all_entries if robots.can_fetch(e.url)]
            blocked = pre_count - len(all_entries)
            if blocked:
                logger.info(f"robots.txt blocked {blocked}/{pre_count} sitemap URLs")

        if all_entries:
            # ── Score + select (strict budget, separate PDFs from pages) ───────
            selected_pages, selected_pdfs = select_pages(
                all_entries, max_pages=MAX_PAGES_PER_SCHOOL
            )

            # ── Crawl ──────────────────────────────────────────────────────────
            spider = SchoolSpider(
                start_url=url,
                school_name=school_name,
                selected_pages=[e.url for e in selected_pages],
                selected_pdfs=[e.url for e in selected_pdfs],
                crawl_dir=crawl_dir,
                robots=robots,
                use_js=use_js,
            )
            result = spider.run()
            result.sitemap_urls_found    = len(all_entries) + blocked
            result.sitemap_urls_selected = len(selected_pages) + len(selected_pdfs)
            return result

        logger.warning("Sitemap found but contained no usable URLs — falling back")

    # ── Fallback: no sitemap or empty sitemap ──────────────────────────────────
    logger.warning("Using fallback link-following crawl (no sitemap)")
    spider = FallbackSpider(
        start_url=url,
        school_name=school_name,
        crawl_dir=crawl_dir,
        robots=robots,
    )
    return spider.run()
