"""
extractor.py — Extract structured school data using LangExtract + Gemini

Takes raw text (from web pages or PDFs) and extracts:
  - Vendors / suppliers + contract details + expiry dates
  - Budgets
  - Projects
  - Problems / issues
  - Board members
  - Contractors

Each extraction is grounded to its exact source location in the document.
"""

import json
import textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    GEMINI_API_KEY,
    LANGEXTRACT_MODEL,
    OUTPUT_DIR,
    RETRY_ATTEMPTS,
    RETRY_WAIT_MIN,
    RETRY_WAIT_MAX,
)


# ── Output data classes ───────────────────────────────────────────────────────

@dataclass
class ExtractedEntity:
    """A single extracted entity with its source grounding."""
    entity_type: str
    text: str
    attributes: dict
    source_url: str
    source_type: str
    source_page: Optional[int] = None
    school_name: str = ""
    domain: str = ""


@dataclass
class ExtractionResult:
    source_url: str
    source_type: str
    entities: list[ExtractedEntity] = field(default_factory=list)


# ── LangExtract prompt & examples ────────────────────────────────────────────

EXTRACTION_PROMPT = textwrap.dedent("""\
    You are an expert analyst extracting structured intelligence from school
    websites, board meeting documents, financial reports, and procurement pages.

    IMPORTANT RULES:
    - Extract ONLY meaningful entities with real information.  Do NOT extract
      brand names that appear in navigation bars, footers, cookie banners, 
      or boilerplate website chrome (e.g. "Powered by SchoolPointe").
    - For VENDOR entities: only extract if there is evidence the school USES,
      PAYS, or CONTRACTS with the company.  A mere mention or link is not enough.
      Include the FULL surrounding context as extraction_text so we capture
      what service is provided and any contract/cost details.
    - When a vendor is mentioned with its service description, ALWAYS fill in
      service_type with what the vendor does for the school.
    - Prefer extracting LONGER extraction_text passages (2-4 sentences) that
      include context, rather than just the entity name alone.
    - Leave attribute values as empty string "" only when the information is
      truly not present in the source text.
    - Do NOT invent or hallucinate information.  Use exact text from the source.

    Entity types to extract:

    1. VENDOR — a company, platform, or service provider the school actively
       uses, pays, or has contracted with.
       Attributes: vendor_name, service_type, contract_value, expiry_date, status
       Examples of service_type: "student information system", "learning management
       system", "food service provider", "IT support", "curriculum provider",
       "assessment platform", "fundraising platform", "transportation",
       "custodial services", "consulting", "professional development"

    2. BUDGET — any budget allocation, financial figure, grant, revenue, or
       expenditure mentioned with a dollar amount or percentage.
       Attributes: amount, currency, category, period, funding_source, status

    3. PROJECT — any capital improvement, construction, renovation, technology
       rollout, or operational initiative with a defined scope.
       Attributes: project_name, description, value, timeline, status, vendor

    4. PROBLEM — any issue, complaint, risk, concern, audit finding, or
       deficiency raised in the document.
       Attributes: description, category, severity, date_mentioned, resolution

    5. BOARD_MEMBER — any named individual serving on the board of trustees,
       school board, governing body, or in a leadership/governance role.
       Attributes: name, role, term_start, term_end

    6. CONTRACTOR — any construction company, trades firm, or specialist
       contractor hired for a specific project or maintenance work.
       Attributes: contractor_name, trade, project, contract_value, expiry_date

    SKIP these (do NOT extract):
    - Navigation menu items, page titles, breadcrumbs
    - Generic website features ("Login", "Contact Us", "Search")
    - Social media links or sharing buttons
    - Cookie consent / privacy policy boilerplate
    - Vendor names that only appear in footer credits or "Powered by" text
    - Duplicate mentions of the same entity on the same page

    Extract entities in order of appearance.  For each entity, include enough
    surrounding text in extraction_text to understand the CONTEXT — why is this
    vendor/budget/project mentioned?  What is its relationship to the school?
""")

EXTRACTION_EXAMPLES = [
    {
        "text": "The board approved a contract with Oracle New Zealand Ltd for student management software valued at $48,500 + GST, expiring 30 June 2026.",
        "extractions": [
            {
                "extraction_class": "vendor",
                "extraction_text": "The board approved a contract with Oracle New Zealand Ltd for student management software valued at $48,500 + GST, expiring 30 June 2026.",
                "attributes": {
                    "vendor_name": "Oracle New Zealand Ltd",
                    "service_type": "student management software",
                    "contract_value": "$48,500 + GST",
                    "expiry_date": "30 June 2026",
                    "status": "approved",
                }
            }
        ]
    },
    {
        "text": "A.C.E. Academy uses Google Workspace for Education to support classroom learning, assignments, communication with teachers, and development of 21st-century digital skills. This platform includes tools such as Gmail, Google Docs, and Google Classroom.",
        "extractions": [
            {
                "extraction_class": "vendor",
                "extraction_text": "A.C.E. Academy uses Google Workspace for Education to support classroom learning, assignments, communication with teachers, and development of 21st-century digital skills. This platform includes tools such as Gmail, Google Docs, and Google Classroom.",
                "attributes": {
                    "vendor_name": "Google",
                    "service_type": "Google Workspace for Education — classroom learning platform including Gmail, Google Docs, Google Classroom for assignments, communication, and digital skills",
                    "contract_value": "",
                    "expiry_date": "",
                    "status": "active",
                }
            }
        ]
    },
    {
        "text": "Infinite Campus is the student information system used for enrollment, attendance, grades, and parent communication.",
        "extractions": [
            {
                "extraction_class": "vendor",
                "extraction_text": "Infinite Campus is the student information system used for enrollment, attendance, grades, and parent communication.",
                "attributes": {
                    "vendor_name": "Infinite Campus",
                    "service_type": "student information system — enrollment, attendance, grades, and parent communication",
                    "contract_value": "",
                    "expiry_date": "",
                    "status": "active",
                }
            }
        ]
    },
    {
        "text": "Motion carried: Capital works budget of $350,000 allocated from Ministry of Education funding for roof replacement at Block B.",
        "extractions": [
            {
                "extraction_class": "budget",
                "extraction_text": "Capital works budget of $350,000 allocated from Ministry of Education funding for roof replacement at Block B.",
                "attributes": {
                    "amount": "350000",
                    "currency": "USD",
                    "category": "capital works",
                    "period": "",
                    "funding_source": "Ministry of Education",
                    "status": "approved",
                }
            },
            {
                "extraction_class": "project",
                "extraction_text": "roof replacement at Block B, budget of $350,000 allocated from Ministry of Education funding",
                "attributes": {
                    "project_name": "Block B Roof Replacement",
                    "description": "Roof replacement at Block B",
                    "value": "$350,000",
                    "timeline": "",
                    "status": "approved",
                    "vendor": "",
                }
            }
        ]
    },
    {
        "text": "Principal reported ongoing leaking in the gymnasium changing rooms. Issue has been ongoing for 3 months and has been escalated to the property manager.",
        "extractions": [
            {
                "extraction_class": "problem",
                "extraction_text": "Principal reported ongoing leaking in the gymnasium changing rooms. Issue has been ongoing for 3 months and has been escalated to the property manager.",
                "attributes": {
                    "description": "Ongoing leaking in gymnasium changing rooms, persisting for 3 months",
                    "category": "infrastructure / maintenance",
                    "severity": "medium",
                    "date_mentioned": "",
                    "resolution": "escalated to property manager",
                }
            }
        ]
    },
    {
        "text": "Chairperson: Sarah Mitchell (re-elected for 3-year term commencing January 2024). Board member: James Taufa.",
        "extractions": [
            {
                "extraction_class": "board_member",
                "extraction_text": "Chairperson: Sarah Mitchell (re-elected for 3-year term commencing January 2024)",
                "attributes": {
                    "name": "Sarah Mitchell",
                    "role": "Chairperson",
                    "term_start": "January 2024",
                    "term_end": "January 2027",
                }
            },
            {
                "extraction_class": "board_member",
                "extraction_text": "Board member: James Taufa",
                "attributes": {
                    "name": "James Taufa",
                    "role": "Board member",
                    "term_start": "",
                    "term_end": "",
                }
            }
        ]
    },
    {
        "text": "The school uses Wit & Wisdom for Kindergarten through 8th grade ELA instruction and National Training Network (NTN) for math instruction across all grade levels.",
        "extractions": [
            {
                "extraction_class": "vendor",
                "extraction_text": "The school uses Wit & Wisdom for Kindergarten through 8th grade ELA instruction",
                "attributes": {
                    "vendor_name": "Wit & Wisdom",
                    "service_type": "ELA curriculum provider for K-8th grade",
                    "contract_value": "",
                    "expiry_date": "",
                    "status": "active",
                }
            },
            {
                "extraction_class": "vendor",
                "extraction_text": "National Training Network (NTN) for math instruction across all grade levels",
                "attributes": {
                    "vendor_name": "National Training Network (NTN)",
                    "service_type": "math curriculum / instruction provider for all grade levels",
                    "contract_value": "",
                    "expiry_date": "",
                    "status": "active",
                }
            }
        ]
    },
]


# ── Confidence filter ─────────────────────────────────────────────────────────

_MIN_UNIQUE_TOKENS = 3  # extraction_text must have ≥3 unique tokens beyond type label


def _filter_low_confidence(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    """
    Drop entities whose extraction_text is too thin to be meaningful.

    If the extraction_text has fewer than _MIN_UNIQUE_TOKENS unique tokens
    after removing the entity type label words, the entity is almost
    certainly noise (e.g. a bare brand name from a footer).
    """
    import re
    kept, dropped = [], 0
    for e in entities:
        text = e.text.lower()
        # Remove the entity type label words from the token set
        label_words = set(e.entity_type.lower().replace("_", " ").split())
        tokens = set(re.findall(r"\b[a-z]{2,}\b", text)) - label_words
        if len(tokens) >= _MIN_UNIQUE_TOKENS:
            kept.append(e)
        else:
            dropped += 1
    if dropped:
        logger.info(f"Confidence filter: dropped {dropped} thin entities")
    return kept


# ── Extractor class ───────────────────────────────────────────────────────────

class SchoolDataExtractor:
    """Wraps LangExtract to extract structured school entities from text."""

    def __init__(self):
        self._lx = None
        self._examples = None
        self._model_id = LANGEXTRACT_MODEL
        self._setup()

    def _setup(self):
        try:
            import langextract as lx

            examples = []
            for ex in EXTRACTION_EXAMPLES:
                extractions = [
                    lx.data.Extraction(
                        extraction_class=e["extraction_class"],
                        extraction_text=e["extraction_text"],
                        attributes=e["attributes"],
                    )
                    for e in ex["extractions"]
                ]
                examples.append(lx.data.ExampleData(
                    text=ex["text"],
                    extractions=extractions,
                ))

            self._lx = lx
            self._examples = examples
            logger.info(f"LangExtract ready | model: {self._model_id}")

        except ImportError:
            logger.error("LangExtract not installed. Run: pip install langextract")
            raise

    def extract_from_text(
        self,
        text: str,
        source_url: str,
        source_type: str,
        school_name: str,
        domain: str,
        source_page: Optional[int] = None,
        max_workers: int = 4,
    ) -> ExtractionResult:
        """Run LangExtract on a block of text and return structured entities."""

        if not text or len(text.strip()) < 100:
            return ExtractionResult(source_url=source_url, source_type=source_type)

        result = ExtractionResult(source_url=source_url, source_type=source_type)

        try:
            lx = self._lx

            raw_dir = OUTPUT_DIR / "langextract_raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            extraction_result = self._call_langextract(text)

            if hasattr(extraction_result, "extractions"):
                for ext in extraction_result.extractions:
                    result.entities.append(ExtractedEntity(
                        entity_type=ext.extraction_class,
                        text=ext.extraction_text,
                        attributes=ext.attributes or {},
                        source_url=source_url,
                        source_type=source_type,
                        source_page=source_page,
                        school_name=school_name,
                        domain=domain,
                    ))

            # Drop entities with too-thin extraction text (noise filter)
            result.entities = _filter_low_confidence(result.entities)

            logger.info(
                f"Extracted {len(result.entities)} entities from {source_url[:60]}"
                + (f" p.{source_page}" if source_page else "")
            )

        except Exception as e:
            logger.error(f"LangExtract failed on {source_url}: {e}")

        return result

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
        reraise=True,
    )
    def _call_langextract(self, text: str):
        """Isolated LangExtract API call with automatic retry on transient errors."""
        return self._lx.extract(
            text_or_documents=text,
            prompt_description=EXTRACTION_PROMPT,
            examples=self._examples,
            model_id=self._model_id,
            api_key=GEMINI_API_KEY or None,
            max_workers=4,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_extractor: Optional[SchoolDataExtractor] = None

def get_extractor() -> SchoolDataExtractor:
    global _extractor
    if _extractor is None:
        _extractor = SchoolDataExtractor()
    return _extractor
