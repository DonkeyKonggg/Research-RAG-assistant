# AI Research Assistant

A lightweight Retrieval-Augmented Generation (RAG) tool that lets researchers
query a collection of scientific PDFs and receive cited, grounded answers.

---

## What is RAG?

**Retrieval-Augmented Generation** combines two ideas:

1. **Retrieval** – find the passages in your documents that are most relevant
   to a question (using vector similarity search).
2. **Generation** – feed those passages to a Large Language Model (LLM) and ask
   it to produce an answer based *only* on what was retrieved.

Because the LLM sees the actual source text, it can answer accurately without
"hallucinating" facts, and it can tell you exactly which passages support each
claim.

---

## How the pipeline works

```
PDF files
   │
   ▼
Text extraction (PyPDF)
   │   Extract raw text page by page
   ▼
Chunking
   │   Split text into ~600-token overlapping segments
   ▼
Embedding (OpenAI text-embedding-3-small)
   │   Convert each chunk into a high-dimensional vector
   ▼
Vector database (FAISS)
   │   Index all vectors for fast similarity search
   ▼
User question
   │
   ├─► Embed question → query vector
   │
   ▼
Retrieval
   │   Find top-k most similar chunks (cosine similarity)
   ▼
LLM answer generation (GPT-4o-mini)
   │   Prompt: "Answer using only these passages. Cite sources."
   ▼
Answer + cited excerpts displayed in Streamlit UI
```

---

## Project structure

```
ai-research-rag/
├── app.py            # Streamlit web interface
├── rag_pipeline.py   # Core RAG logic (load → chunk → embed → retrieve → answer)
├── requirements.txt  # Python dependencies
├── papers/           # Drop your PDF files here
│   └── example_paper.pdf
└── README.md
```

---

## Quick start

### 1. Clone and install

```bash
git clone <repo-url>
cd ai-research-rag

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Add your papers

Copy one or more PDF files into the `papers/` directory:

```
papers/
  attention_is_all_you_need.pdf
  bert_paper.pdf
  ...
```

### 3. Set your OpenAI API key

You can either export it as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

Or enter it in the sidebar when the app starts.

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Example questions

Depending on the papers you load, you might ask:

- *What datasets were used to evaluate the model?*
- *What are the main limitations described by the authors?*
- *How does this approach compare to previous methods?*
- *What activation function was used in the feed-forward layers?*
- *What future work do the authors suggest?*

---

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 600 tokens | Target chunk size (in `rag_pipeline.py`) |
| `CHUNK_OVERLAP` | 100 tokens | Overlap between consecutive chunks |
| `TOP_K` | 4 | Number of retrieved passages per query |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model for answer generation |

All constants are at the top of `rag_pipeline.py` and can be changed freely.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `openai` | Embeddings + LLM |
| `faiss-cpu` | Vector similarity search |
| `pypdf` | PDF text extraction |
| `numpy` | Vector math |

---

## How sources are displayed

After each answer the UI shows the retrieved passages that informed it:

```
Answer
[generated answer with inline citations like [Source 1]]

Sources used:

  Source 1 — paper_name.pdf  (page 4, relevance 87.3%)
  ┌─────────────────────────────────────────────────────┐
  │ Paper: paper_name.pdf                               │
  │ Page: 4                                             │
  │ Relevance score: 87.3%                              │
  │ Excerpt:                                            │
  │ > "…retrieved paragraph text…"                      │
  └─────────────────────────────────────────────────────┘
```

---

## Notes

- The index is built **once per session** and cached. To re-index after adding
  new PDFs, restart the Streamlit app (or press **R** to hard-rerun).
- Only text-based PDFs are supported. Scanned image PDFs require an OCR step
  (e.g. `pytesseract`) before they can be processed.
- Costs are minimal: embedding 50 pages costs roughly $0.001 with
  `text-embedding-3-small`; each query costs a few cents with `gpt-4o-mini`.
