"""
rag_pipeline.py

Core RAG pipeline with two pluggable embedding backends:
  - OpenAIEmbedder  : uses text-embedding-3-small via API (requires internet)
  - LocalEmbedder   : uses sentence-transformers all-MiniLM-L6-v2 (fully offline)

LLM answer generation uses OpenAI gpt-4o-mini when a client is provided;
otherwise the top retrieved passages are formatted as a structured answer
(retrieval-only mode, works 100% offline).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

import faiss
import httpx
import numpy as np
from openai import OpenAI, APIConnectionError, APIStatusError
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PAPERS_DIR = Path("papers")
CHUNK_SIZE = 600        # approximate tokens per chunk (chars ÷ 4 ≈ tokens)
CHUNK_OVERLAP = 100     # overlap between consecutive chunks (in chars)
TOP_K = 4               # number of retrieved chunks
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"  # ~90 MB, downloads on first use

EMBED_BATCH_SIZE = 50   # max chunks per OpenAI embedding call
API_TIMEOUT = 60        # seconds before a single request times out
MAX_RETRIES = 3         # number of retries on transient API errors

CHAR_CHUNK_SIZE = CHUNK_SIZE * 4       # ~2400 chars
CHAR_CHUNK_OVERLAP = CHUNK_OVERLAP * 4  # ~400 chars


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    source: str   # PDF filename
    page: int


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity (higher = more relevant)


@dataclass
class RAGAnswer:
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Embedder protocol + implementations
# ---------------------------------------------------------------------------

@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return float32 array of shape (n, dim)."""
        ...


class OpenAIEmbedder:
    """Embeds text via the OpenAI API (requires internet + API key)."""

    def __init__(self, client: OpenAI):
        self.client = client

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        delay = 2.0
        last_exc: Exception = RuntimeError("unknown error")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.embeddings.create(
                    model=EMBEDDING_MODEL, input=texts
                )
                return [item.embedding for item in response.data]
            except (APIConnectionError, httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    print(
                        f"  Connection error on attempt {attempt}/{MAX_RETRIES}, "
                        f"retrying in {delay:.0f}s…"
                    )
                    time.sleep(delay)
                    delay *= 2
            except APIStatusError as exc:
                if exc.status_code == 429 and attempt < MAX_RETRIES:
                    print(f"  Rate limited, retrying in {delay:.0f}s…")
                    time.sleep(delay)
                    delay *= 2
                    last_exc = exc
                else:
                    raise
        raise last_exc

    def embed(self, texts: List[str]) -> np.ndarray:
        all_vectors: List[List[float]] = []
        total = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
        for i, start in enumerate(range(0, len(texts), EMBED_BATCH_SIZE), 1):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            print(f"  Embedding batch {i}/{total} ({len(batch)} chunks)…")
            all_vectors.extend(self._embed_batch(batch))
        return np.array(all_vectors, dtype=np.float32)


class LocalEmbedder:
    """
    Embeds text locally using sentence-transformers (no internet after first run).
    The model (~90 MB) is downloaded once and cached by HuggingFace.
    """

    def __init__(self, model_name: str = LOCAL_EMBED_MODEL):
        # Import lazily so the OpenAI-only path doesn't pay the import cost
        from sentence_transformers import SentenceTransformer  # type: ignore
        print(f"  Loading local embedding model '{model_name}'…")
        self._model = SentenceTransformer(model_name)
        print("  Local model ready.")

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(texts, show_progress_bar=True, batch_size=64)
        return np.array(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# PDF loading & text extraction
# ---------------------------------------------------------------------------

def load_pdfs(papers_dir: Path = PAPERS_DIR) -> List[Chunk]:
    """Read every PDF in *papers_dir* and return page-level Chunk objects."""
    chunks: List[Chunk] = []
    pdf_files = sorted(papers_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{papers_dir.resolve()}'. "
            "Please add at least one PDF to the papers/ folder."
        )

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                chunks.append(Chunk(text=text, source=pdf_path.name, page=page_num))

    return chunks


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def split_into_chunks(page_chunks: List[Chunk]) -> List[Chunk]:
    """Split page-level chunks into overlapping ~CHUNK_SIZE-token segments."""
    result: List[Chunk] = []

    for page_chunk in page_chunks:
        text = page_chunk.text
        start = 0

        while start < len(text):
            end = start + CHAR_CHUNK_SIZE
            segment = text[start:end]

            if end < len(text):
                last_period = segment.rfind(". ")
                if last_period != -1 and last_period > CHAR_CHUNK_SIZE // 2:
                    segment = segment[: last_period + 1]

            segment = segment.strip()
            if segment:
                result.append(
                    Chunk(text=segment, source=page_chunk.source, page=page_chunk.page)
                )

            step = len(segment) - CHAR_CHUNK_OVERLAP
            start += step if step > 0 else max(1, len(segment))

    return result


# ---------------------------------------------------------------------------
# FAISS vector store
# ---------------------------------------------------------------------------

class VectorStore:
    """Thin wrapper around a FAISS flat inner-product (cosine) index."""

    def __init__(self, chunks: List[Chunk], embeddings: np.ndarray):
        self.chunks = chunks
        dim = embeddings.shape[1]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.where(norms == 0, 1, norms)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(normalized)

    def search(self, query_embedding: np.ndarray, k: int = TOP_K) -> List[RetrievedChunk]:
        norm = np.linalg.norm(query_embedding)
        q = (query_embedding / norm if norm > 0 else query_embedding).reshape(1, -1)
        scores, indices = self.index.search(q, k)
        return [
            RetrievedChunk(chunk=self.chunks[idx], score=float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(
    embedder: Embedder,
    papers_dir: Path = PAPERS_DIR,
) -> VectorStore:
    """
    Load PDFs → chunk → embed → FAISS index.
    *embedder* can be an OpenAIEmbedder or a LocalEmbedder.
    """
    print("Loading PDFs…")
    page_chunks = load_pdfs(papers_dir)
    sources = len(set(c.source for c in page_chunks))
    print(f"  Loaded {len(page_chunks)} pages from {sources} PDF(s).")

    print("Splitting into chunks…")
    chunks = split_into_chunks(page_chunks)
    print(f"  Created {len(chunks)} text chunks.")

    print("Generating embeddings…")
    embeddings = embedder.embed([c.text for c in chunks])
    print(f"  Embedded {len(embeddings)} chunks (dim={embeddings.shape[1]}).")

    print("Building FAISS index…")
    store = VectorStore(chunks, embeddings)
    print("  Index ready.\n")

    return store


# ---------------------------------------------------------------------------
# Query & answer generation
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    openai_client: Optional[OpenAI] = None,
    top_k: int = TOP_K,
) -> RAGAnswer:
    """
    Retrieve the top-k chunks for *question*, then either:
      - Call the OpenAI LLM to synthesize a cited answer (if openai_client is set), or
      - Format the retrieved passages directly as the answer (offline/retrieval-only mode).
    """
    q_embedding = embedder.embed([question])[0]
    retrieved = store.search(q_embedding, k=top_k)

    if not retrieved:
        return RAGAnswer(answer="No relevant passages found.", sources=[])

    # --- LLM synthesis (cloud mode) ---
    if openai_client is not None:
        context_parts = [
            f"[Source {i}: {r.chunk.source}, page {r.chunk.page}]\n{r.chunk.text}"
            for i, r in enumerate(retrieved, 1)
        ]
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are a scientific research assistant. "
            "Answer the user's question using ONLY the provided context passages. "
            "If the context does not contain enough information, say so. "
            "Be concise and precise. Cite the source numbers (e.g. [Source 1]) inline."
        )

        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        answer_text = response.choices[0].message.content.strip()

    # --- Retrieval-only (offline mode) ---
    else:
        lines = [
            "**Retrieval-only mode** — no LLM synthesis (OpenAI API not configured).\n",
            "The most relevant passages found for your question are shown below as sources.",
        ]
        answer_text = "\n".join(lines)

    return RAGAnswer(answer=answer_text, sources=retrieved)
