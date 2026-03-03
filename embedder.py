"""
embedder.py — Pluggable embedding module

Default: Google Gemini text-embedding-004 (768 dims) — same API key as extractor.
Fallback options: OpenAI, HuggingFace (set EMBEDDING_PROVIDER in .env)
"""

from typing import Optional
from abc import ABC, abstractmethod

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import (
    EMBEDDING_PROVIDER, EMBEDDING_MODEL, GEMINI_API_KEY,
    RETRY_ATTEMPTS, RETRY_WAIT_MIN, RETRY_WAIT_MAX,
)
from chunker import Chunk


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

        for chunk, vector in zip(chunks, all_vectors):
            chunk.vector = vector

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
