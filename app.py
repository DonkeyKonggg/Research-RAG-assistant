"""
app.py

Streamlit UI for the AI Research RAG Assistant.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st

from rag_pipeline import RAGAnswer, VectorStore, build_index, answer_question
from openai import OpenAI
from pathlib import Path

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Session-state helpers (cache the index across re-runs)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_index() -> tuple[VectorStore, OpenAI]:
    """Build (or reload) the FAISS index once per session."""
    return build_index()


# ---------------------------------------------------------------------------
# Sidebar – configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-…",
        help="Your key is never stored. It is only used for this session.",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    st.markdown("---")
    st.subheader("Papers folder")
    papers_path = Path("papers")
    pdf_files = sorted(papers_path.glob("*.pdf")) if papers_path.exists() else []

    if pdf_files:
        st.success(f"{len(pdf_files)} PDF(s) loaded")
        for f in pdf_files:
            st.caption(f"📄 {f.name}")
    else:
        st.warning("No PDFs found in `papers/`. Add `.pdf` files and restart the app.")

    st.markdown("---")
    top_k = st.slider("Retrieved passages (top-k)", min_value=1, max_value=8, value=4)
    st.caption("How many text chunks are retrieved before the LLM generates an answer.")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🔬 AI Research Assistant")
st.markdown(
    "Ask a question about the papers in your `papers/` folder. "
    "The assistant retrieves the most relevant passages and generates "
    "a cited answer using only what it finds."
)

st.markdown("---")

# Question input
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

    elif not os.environ.get("OPENAI_API_KEY"):
        st.error("Please enter your OpenAI API key in the sidebar first.")

    elif not pdf_files:
        st.error("No PDFs found in the `papers/` folder. Please add at least one PDF.")

    else:
        with st.spinner("Indexing papers and retrieving relevant passages…"):
            try:
                store, client = get_index()
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()
            except EnvironmentError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"Failed to build the index: {e}")
                st.stop()

        with st.spinner("Generating answer…"):
            try:
                result: RAGAnswer = answer_question(question, store, client, top_k=top_k)
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                st.stop()

        # ----------------------------------------------------------------
        # Display: Answer
        # ----------------------------------------------------------------
        st.markdown("## Answer")
        st.markdown(result.answer)

        # ----------------------------------------------------------------
        # Display: Sources
        # ----------------------------------------------------------------
        st.markdown("---")
        st.markdown("## Sources used")

        if not result.sources:
            st.info("No sources retrieved.")
        else:
            for i, retrieved in enumerate(result.sources, start=1):
                chunk = retrieved.chunk
                similarity_pct = f"{retrieved.score * 100:.1f}%"

                with st.expander(
                    f"Source {i} — {chunk.source}  (page {chunk.page}, relevance {similarity_pct})",
                    expanded=True,
                ):
                    st.markdown(f"**Paper:** `{chunk.source}`")
                    st.markdown(f"**Page:** {chunk.page}")
                    st.markdown(f"**Relevance score:** {similarity_pct}")
                    st.markdown("**Excerpt:**")
                    # Show at most 600 chars for readability
                    excerpt = chunk.text[:600]
                    if len(chunk.text) > 600:
                        excerpt += "…"
                    st.markdown(f"> {excerpt}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Built with Streamlit · LangChain-free RAG · FAISS · OpenAI · PyPDF  |  "
    "Drop PDFs into `papers/` and refresh to re-index."
)
