"""
embedder.py — Pluggable embedding module

Default: Google Gemini text-embedding-004 (768 dims) — same API key as extractor.
Fallback options: OpenAI, HuggingFace (set EMBEDDING_PROVIDER in .env)

Includes SparseVectorizer for hybrid dense+sparse search via Qdrant.
"""

import re
import math
from typing import Optional
from abc import ABC, abstractmethod
from collections import Counter

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import (
    EMBEDDING_PROVIDER, EMBEDDING_MODEL, GEMINI_API_KEY,
    RETRY_ATTEMPTS, RETRY_WAIT_MIN, RETRY_WAIT_MAX,
)
from chunker import Chunk


# ── Sparse vectorizer (BM25-style lexical matching) ──────────────────────────

class SparseVectorizer:
    """
    Simple TF-based sparse vectorizer for hybrid search.

    Tokenises text, removes stopwords, hashes tokens into a fixed-size
    vocabulary space, and returns (indices, values) suitable for Qdrant
    SparseVector.  This gives BM25-style lexical matching alongside
    dense semantic embeddings.
    """

    VOCAB_SIZE = 30_000  # hash buckets — large enough to minimise collisions

    _STOPWORDS = frozenset({
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "this", "that",
        "these", "those", "it", "its", "not", "no", "if", "then", "than",
        "so", "as", "up", "out", "about", "into", "over", "after", "before",
    })

    _TOKEN_RE = re.compile(r"\b[a-z0-9]{2,}\b")

    @classmethod
    def vectorize(cls, text: str) -> tuple[list[int], list[float]]:
        """Return (indices, values) sparse vector for *text*."""
        tokens = cls._TOKEN_RE.findall(text.lower())
        tokens = [t for t in tokens if t not in cls._STOPWORDS]
        if not tokens:
            return [], []

        counts = Counter(tokens)
        sparse: dict[int, float] = {}
        for token, count in counts.items():
            idx = hash(token) % cls.VOCAB_SIZE
            sparse[idx] = sparse.get(idx, 0) + math.log1p(count)

        indices = sorted(sparse.keys())
        values = [sparse[i] for i in indices]
        return indices, values

    @classmethod
    def vectorize_batch(cls, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return [cls.vectorize(t) for t in texts]


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        pass

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> list[Chunk]:
        texts = [c.embed_text for c in chunks]
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_vectors.extend(self.embed(batch))
            logger.info(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")

        # Dense vectors
        for chunk, vector in zip(chunks, all_vectors):
            chunk.vector = vector

        # Sparse vectors (for hybrid search)
        sparse_vecs = SparseVectorizer.vectorize_batch(texts)
        for chunk, (indices, values) in zip(chunks, sparse_vecs):
            chunk.sparse_indices = indices
            chunk.sparse_values = values

        return chunks


# ── Gemini (default) ──────────────────────────────────────────────────────────

class GeminiEmbedder(BaseEmbedder):
    def __init__(self, model: str = EMBEDDING_MODEL):
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self.genai = genai
            self.model = model
            logger.info(f"Gemini embedder ready | model: {model}")
        except ImportError:
            raise ImportError("Run: pip install google-generativeai")

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
        reraise=True,
    )
    def _embed_single_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts with retry. Uses batch API when possible."""
        if len(texts) == 1:
            result = self.genai.embed_content(
                model=f"models/{self.model}",
                content=texts[0],
                task_type="retrieval_document",
            )
            return [result["embedding"]]

        # Batch embed: embed_content supports a list of content strings
        result = self.genai.embed_content(
            model=f"models/{self.model}",
            content=texts,
            task_type="retrieval_document",
        )
        return result["embedding"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed_single_batch(texts)


# ── OpenAI (fallback) ─────────────────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = EMBEDDING_MODEL):
        try:
            import os
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            self.model  = model
            logger.info(f"OpenAI embedder ready | model: {model}")
        except ImportError:
            raise ImportError("Run: pip install openai")

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
        reraise=True,
    )
    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


# ── HuggingFace (free local — GPU accelerated) ───────────────────────────────

# Models that require a query instruction prefix for retrieval
_BGE_MODELS = {"bge-large-en-v1.5", "bge-base-en-v1.5", "bge-small-en-v1.5"}


class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, model: str = "BAAI/bge-large-en-v1.5"):
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading HuggingFace model: {model} on {device}...")
            self.model = SentenceTransformer(model, device=device)
            self._is_bge = any(bge in model for bge in _BGE_MODELS)
            self._device = device
            logger.info(
                f"HuggingFace embedder ready | {model} | {device} | "
                f"dim={self.model.get_sentence_embedding_dimension()}"
            )
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers torch")

    def embed(self, texts: list[str]) -> list[list[float]]:
        # BGE models need "Represent this sentence: " prefix for documents
        if self._is_bge:
            texts = [f"Represent this sentence: {t}" for t in texts]
        return self.model.encode(
            texts, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).tolist()


# ── Factory ───────────────────────────────────────────────────────────────────

def get_embedder(provider: Optional[str] = None) -> BaseEmbedder:
    p = (provider or EMBEDDING_PROVIDER).lower()
    if p == "gemini":
        return GeminiEmbedder()
    elif p == "openai":
        return OpenAIEmbedder()
    elif p == "huggingface":
        return HuggingFaceEmbedder(model=EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unknown provider: '{p}'. Choose: gemini | openai | huggingface")
