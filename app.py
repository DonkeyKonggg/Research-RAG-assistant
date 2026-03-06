"""
app.py

Streamlit UI for the AI Research RAG Assistant.

Two modes:
  - OpenAI (cloud): OpenAI embeddings + GPT-4o-mini answer synthesis
  - Local  (offline): sentence-transformers embeddings, no LLM required

Run with:
    streamlit run app.py
"""

import os
from pathlib import Path
import httpx
import streamlit as st

from openai import OpenAI

from rag_pipeline import (
    RAGAnswer,
    VectorStore,
    Embedder,
    OpenAIEmbedder,
    LocalEmbedder,
    build_index,
    answer_question,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Cached index builder  (keyed on provider + api_key so switching rebuilds)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_index(provider: str, api_key: str) -> tuple[VectorStore, Embedder, OpenAI | None]:
    """
    Build the FAISS index once per (provider, api_key) combination.
    Returns (store, embedder, openai_client_or_None).
    """
    if provider == "OpenAI (cloud)":
        if not api_key:
            raise EnvironmentError(
                "Please enter your OpenAI API key in the sidebar."
            )
        client = OpenAI(api_key=api_key, timeout=60, max_retries=0, http_client=httpx.Client(verify=False))
        embedder: Embedder = OpenAIEmbedder(client)
        store = build_index(embedder)
        return store, embedder, client

    else:  # Local / Offline
        embedder = LocalEmbedder()
        store = build_index(embedder)
        return store, embedder, None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    provider = st.radio(
        "Embedding & LLM provider",
        ["OpenAI (cloud)", "Local / Offline"],
        help=(
            "**OpenAI (cloud):** uses text-embedding-3-small + GPT-4o-mini. "
            "Requires an API key and internet access.\n\n"
            "**Local / Offline:** uses sentence-transformers all-MiniLM-L6-v2 "
            "running on your machine. No API key or internet needed after the "
            "first run (model ~90 MB, downloaded once)."
        ),
    )

    api_key_input = ""
    if provider == "OpenAI (cloud)":
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-…",
            help="Used only for this session, never stored.",
        )
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
    else:
        st.info(
            "Running fully offline.\n\n"
            "Embeddings are computed locally. "
            "Retrieved passages are shown directly — no LLM synthesis."
        )

    st.markdown("---")
    st.subheader("Papers folder")
    papers_path = Path("papers")
    pdf_files = sorted(papers_path.glob("*.pdf")) if papers_path.exists() else []

    if pdf_files:
        st.success(f"{len(pdf_files)} PDF(s) loaded")
        for f in pdf_files:
            st.caption(f"📄 {f.name}")
    else:
        st.warning("No PDFs found in `papers/`. Add `.pdf` files and restart.")

    st.markdown("---")
    top_k = st.slider("Retrieved passages (top-k)", min_value=1, max_value=8, value=4)
    st.caption("How many text chunks are retrieved before generating an answer.")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🔬 AI Research Assistant")
st.markdown(
    "Ask a question about the papers in your `papers/` folder. "
    "The assistant retrieves the most relevant passages and generates "
    "a cited answer using only what it finds."
)

if provider == "Local / Offline":
    st.info(
        "**Local mode active.** "
        "The top retrieved passages are shown as your answer — "
        "no LLM synthesis. Switch to *OpenAI (cloud)* in the sidebar for "
        "generated answers."
    )

st.markdown("---")

question = st.text_area(
    "Your research question",
    placeholder="e.g. What methods were used to evaluate model performance?",
    height=90,
)

ask_button = st.button("Ask", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

if ask_button:
    question = question.strip()

    if not question:
        st.warning("Please enter a question before clicking Ask.")

    elif provider == "OpenAI (cloud)" and not (
        api_key_input or os.environ.get("OPENAI_API_KEY")
    ):
        st.error("Please enter your OpenAI API key in the sidebar.")

    elif not pdf_files:
        st.error("No PDFs found in `papers/`. Please add at least one PDF.")

    else:
        effective_key = api_key_input or os.environ.get("OPENAI_API_KEY", "")

        with st.spinner("Indexing papers… (first run may take a moment)"):
            try:
                store, embedder, openai_client = get_index(provider, effective_key)
            except EnvironmentError as e:
                st.error(str(e))
                st.stop()
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                err = str(e)
                if any(w in err.lower() for w in ("connect", "network", "timeout")):
                    st.error(
                        f"**Connection error contacting the OpenAI API.**\n\n"
                        f"`{err}`\n\n"
                        "Switch to **Local / Offline** mode in the sidebar to "
                        "run without internet access."
                    )
                else:
                    st.error(f"Failed to build the index: {e}")
                st.stop()

        with st.spinner("Retrieving passages and generating answer…"):
            try:
                result: RAGAnswer = answer_question(
                    question, store, embedder, openai_client, top_k=top_k
                )
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                st.stop()

        # ----------------------------------------------------------------
        # Answer
        # ----------------------------------------------------------------
        st.markdown("## Answer")
        st.markdown(result.answer)

        # ----------------------------------------------------------------
        # Sources
        # ----------------------------------------------------------------
        st.markdown("---")
        st.markdown("## Sources used")

        if not result.sources:
            st.info("No sources retrieved.")
        else:
            for i, retrieved in enumerate(result.sources, start=1):
                chunk = retrieved.chunk
                relevance = f"{retrieved.score * 100:.1f}%"

                with st.expander(
                    f"Source {i} — {chunk.source}  (page {chunk.page}, relevance {relevance})",
                    expanded=True,
                ):
                    st.markdown(f"**Paper:** `{chunk.source}`")
                    st.markdown(f"**Page:** {chunk.page}")
                    st.markdown(f"**Relevance score:** {relevance}")
                    st.markdown("**Excerpt:**")
                    excerpt = chunk.text[:600] + ("…" if len(chunk.text) > 600 else "")
                    st.markdown(f"> {excerpt}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Built with Streamlit · FAISS · PyPDF  |  "
    "Cloud mode: OpenAI embeddings + GPT-4o-mini  |  "
    "Offline mode: sentence-transformers all-MiniLM-L6-v2  |  "
    "Drop PDFs into `papers/` and refresh to re-index."
)
