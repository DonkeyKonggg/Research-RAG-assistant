"""
rag_pipeline.py

Core RAG pipeline:
  1. Load PDFs from the /papers folder
  2. Extract and chunk text
  3. Embed chunks with OpenAI embeddings
  4. Store in a FAISS vector index
  5. Retrieve top-k chunks for a query
  6. Generate an answer via an OpenAI LLM
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

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

EMBED_BATCH_SIZE = 50   # max chunks per embedding API call
API_TIMEOUT = 60        # seconds before a single request times out
MAX_RETRIES = 3         # number of retries on transient errors

CHAR_CHUNK_SIZE = CHUNK_SIZE * 4      # ~2400 chars
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
# PDF loading & text extraction
# ---------------------------------------------------------------------------

def load_pdfs(papers_dir: Path = PAPERS_DIR) -> List[Chunk]:
    """Read every PDF in *papers_dir* and return a flat list of page-level chunks."""
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
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                chunks.append(Chunk(text=text, source=pdf_path.name, page=page_num))

    return chunks


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def split_into_chunks(page_chunks: List[Chunk]) -> List[Chunk]:
    """
    Split each page-level chunk into smaller overlapping chunks of
    approximately CHUNK_SIZE tokens.
    """
    result: List[Chunk] = []

    for page_chunk in page_chunks:
        text = page_chunk.text
        start = 0

        while start < len(text):
            end = start + CHAR_CHUNK_SIZE
            segment = text[start:end]

            # Try to break at a sentence boundary
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
            if step <= 0:
                step = max(1, len(segment))
            start += step

    return result


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _embed_batch(texts: List[str], client: OpenAI) -> List[List[float]]:
    """
    Embed a single batch of texts with exponential-backoff retries.
    Raises the last exception if all retries are exhausted.
    """
    delay = 2.0
    last_exc: Exception = RuntimeError("unknown error")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            return [item.embedding for item in response.data]
        except (APIConnectionError, httpx.ConnectError, httpx.TimeoutException) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                print(f"  Connection error on attempt {attempt}/{MAX_RETRIES}, retrying in {delay:.0f}s…")
                time.sleep(delay)
                delay *= 2
        except APIStatusError as exc:
            # 429 rate-limit: back off; other status errors are fatal
            if exc.status_code == 429 and attempt < MAX_RETRIES:
                print(f"  Rate limited, retrying in {delay:.0f}s…")
                time.sleep(delay)
                delay *= 2
                last_exc = exc
            else:
                raise
    raise last_exc


def embed_texts(texts: List[str], client: OpenAI) -> np.ndarray:
    """
    Embed *texts* in batches of EMBED_BATCH_SIZE and return a float32
    array of shape (n, embedding_dim).
    """
    all_vectors: List[List[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        batch_num = start // EMBED_BATCH_SIZE + 1
        total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
        print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)…")
        all_vectors.extend(_embed_batch(batch, client))
    return np.array(all_vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

class VectorStore:
    """Thin wrapper around a FAISS flat inner-product index."""

    def __init__(self, chunks: List[Chunk], embeddings: np.ndarray):
        self.chunks = chunks
        dim = embeddings.shape[1]

        # Normalize so inner product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(normalized)

    def search(self, query_embedding: np.ndarray, k: int = TOP_K) -> List[RetrievedChunk]:
        """Return the *k* most relevant chunks for *query_embedding*."""
        # Normalize query vector
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        query_2d = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_2d, k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score)))

        return results


# ---------------------------------------------------------------------------
# Indexing (build the full pipeline)
# ---------------------------------------------------------------------------

def build_index(papers_dir: Path = PAPERS_DIR) -> tuple[VectorStore, OpenAI]:
    """
    Load PDFs → chunk → embed → build FAISS index.
    Returns *(vector_store, openai_client)*.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please export it before running the app."
        )

    client = OpenAI(api_key=api_key, timeout=API_TIMEOUT, max_retries=0)  # retries managed manually

    print("Loading PDFs…")
    page_chunks = load_pdfs(papers_dir)
    print(f"  Loaded {len(page_chunks)} pages from {len(set(c.source for c in page_chunks))} PDF(s).")

    print("Splitting into chunks…")
    chunks = split_into_chunks(page_chunks)
    print(f"  Created {len(chunks)} text chunks.")

    print("Generating embeddings…")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts, client)
    print(f"  Embedded {len(embeddings)} chunks (dim={embeddings.shape[1]}).")

    print("Building FAISS index…")
    store = VectorStore(chunks, embeddings)
    print("  Index ready.\n")

    return store, client


# ---------------------------------------------------------------------------
# Query & answer generation
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    store: VectorStore,
    client: OpenAI,
    top_k: int = TOP_K,
) -> RAGAnswer:
    """
    Embed *question*, retrieve the top-k chunks, then call the LLM
    to produce a grounded answer.
    """
    # 1. Embed the question
    q_embedding = embed_texts([question], client)[0]

    # 2. Retrieve relevant chunks
    retrieved = store.search(q_embedding, k=top_k)

    if not retrieved:
        return RAGAnswer(answer="No relevant passages found.", sources=[])

    # 3. Build context for the LLM
    context_parts = []
    for i, r in enumerate(retrieved, start=1):
        context_parts.append(
            f"[Source {i}: {r.chunk.source}, page {r.chunk.page}]\n{r.chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a scientific research assistant. "
        "Answer the user's question using ONLY the provided context passages. "
        "If the context does not contain enough information, say so. "
        "Be concise and precise. Cite the source numbers (e.g. [Source 1]) inline."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    # 4. Call the LLM
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )

    answer_text = response.choices[0].message.content.strip()

    return RAGAnswer(answer=answer_text, sources=retrieved)
