"""VeriRAG Streamlit UI.

Run from the project root:

    streamlit run src/verirag/ui/app.py

NOTE: unlike the test scripts, this runs against the REAL persistent stores
configured in settings (chroma_persist_dir, bm25_corpus_path), so anything
uploaded here accumulates in your actual corpus.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import streamlit as st

from verirag.config import settings
from verirag.generation.generator import AnswerFormat
from verirag.graph.build import run_query
from verirag.ingest import ingest_file
from verirag.retrieval.vector_store import list_ingested_documents

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("./data/raw/uploads")
SUPPORTED_TYPES = ["pdf", "docx", "pptx", "txt", "md"]

FORMAT_LABELS: dict[str, AnswerFormat] = {
    "Concise": AnswerFormat.CONCISE,
    "Detailed": AnswerFormat.DETAILED,
}

st.set_page_config(page_title="VeriRAG", page_icon="🔎", layout="wide")


def _save_upload(uploaded_file) -> Path:
    """Write an uploaded file to UPLOAD_DIR, avoiding filename collisions."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    data = uploaded_file.getbuffer()

    target = UPLOAD_DIR / uploaded_file.name
    if target.exists():
        # Same name, possibly different content -- disambiguate with a short
        # content hash rather than silently overwriting the earlier upload.
        digest = hashlib.sha256(bytes(data)).hexdigest()[:8]
        target = target.with_name(f"{target.stem}-{digest}{target.suffix}")

    target.write_bytes(data)
    return target


def _refresh_corpus() -> None:
    """Re-read the corpus listing from the persistent store into session."""
    try:
        st.session_state["corpus"] = list_ingested_documents()
    except Exception as exc:  # noqa: BLE001 - UI must survive a store error
        logger.warning("Could not list corpus: %s", exc)
        st.session_state["corpus"] = []
        st.session_state["corpus_error"] = str(exc)


def _render_sidebar() -> None:
    st.sidebar.header("Corpus")

    uploads = st.sidebar.file_uploader(
        "Upload documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="PDF, DOCX, PPTX, TXT or MD. Figures in PDFs get captioned.",
    )

    if st.sidebar.button("Ingest uploaded files", disabled=not uploads):
        for uploaded_file in uploads or []:
            try:
                saved_path = _save_upload(uploaded_file)
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    result = ingest_file(saved_path)
            except Exception as exc:  # noqa: BLE001 - report, keep going
                logger.exception("Ingestion failed for %s", uploaded_file.name)
                st.sidebar.error(f"{uploaded_file.name}: {exc}")
                continue

            if result.skipped:
                st.sidebar.info(
                    f"{uploaded_file.name}: already in corpus (skipped)"
                )
            else:
                st.sidebar.success(
                    f"{uploaded_file.name}: {result.num_text_chunks} text + "
                    f"{result.num_figure_chunks} figure chunk(s)"
                )
        _refresh_corpus()

    st.sidebar.subheader("Your corpus")
    if "corpus" not in st.session_state:
        _refresh_corpus()

    corpus = st.session_state.get("corpus") or []
    if not corpus:
        st.sidebar.caption("Nothing ingested yet.")
    else:
        st.sidebar.caption(f"{len(corpus)} document(s)")
        for row in corpus:
            icon = "🌐" if row["source_type"] == "external" else "📄"
            name = Path(str(row["source_path"])).name or row["source_path"]
            st.sidebar.markdown(
                f"{icon} **{name}**  \n"
                f"`{row['doc_id'][:20]}` · {row['format']}"
            )

    if st.sidebar.button("Refresh corpus list"):
        _refresh_corpus()


def _render_sources(citations) -> None:
    st.subheader("Sources")
    for citation in citations:
        if citation.source_type == "external":
            st.markdown(
                f"**[{citation.number}]** 🌐 *external* — "
                f"[{citation.url}]({citation.url})"
            )
        else:
            pages = [p for p in citation.pages if p is not None]
            page_text = f" · page(s) {', '.join(map(str, pages))}" if pages else ""
            st.markdown(
                f"**[{citation.number}]** 📄 *local* — "
                f"`{citation.source_path}`{page_text}"
            )


def _render_result(result) -> None:
    if result.abstained:
        st.warning(result.abstention_reason or "VeriRAG abstained.")
    else:
        st.subheader("Answer")
        st.markdown(result.answer or "")
        if result.citations:
            _render_sources(result.citations)

    with st.expander("Show reasoning trace"):
        st.caption(
            f"{len(result.trace)} step(s) · "
            f"{result.regeneration_count} regeneration(s)"
        )
        for i, step in enumerate(result.trace, start=1):
            st.text(f"{i:2}. {step}")


def _render_main() -> None:
    st.title("🔎 VeriRAG")
    st.caption(
        "A self-verifying document QA system with a two-gate verification "
        "critic. It answers from your documents with citations, or abstains."
    )

    query = st.text_input(
        "Question", placeholder="What does the Q4 revenue chart show?"
    )

    left, right = st.columns([1, 2])
    with left:
        format_label = st.selectbox("Answer format", list(FORMAT_LABELS))
    with right:
        corpus_only = st.checkbox(
            "Corpus-only mode",
            value=False,
            help=(
                "When enabled, only your uploaded documents are used - no web "
                "search, and no previously-fetched web content either."
            ),
        )

    if st.button("Ask", type="primary", disabled=not query.strip()):
        with st.spinner("Thinking..."):
            try:
                result = run_query(
                    query.strip(),
                    FORMAT_LABELS[format_label],
                    corpus_only=corpus_only,
                )
            except Exception as exc:  # noqa: BLE001 - surface, do not crash
                logger.exception("Query failed")
                st.error(f"Query failed: {exc}")
                return
        st.session_state["last_result"] = result

    if st.session_state.get("last_result") is not None:
        _render_result(st.session_state["last_result"])


def main() -> None:
    _render_sidebar()
    _render_main()


main()
