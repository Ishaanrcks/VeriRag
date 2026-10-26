"""VeriRAG Streamlit UI.

Run from the project root:

    streamlit run src/verirag/ui/app.py

NOTE: unlike the test scripts, this runs against the REAL persistent stores
configured in settings (chroma_persist_dir, bm25_corpus_path), so anything
uploaded here accumulates in your actual corpus.

This module is presentation only. Every number it shows is read from what the
pipeline already computed -- nothing here re-scores, re-ranks, or re-derives
anything.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

# Local-dev-only: let torch fall back to CPU for ops MPS does not implement,
# instead of raising on an unsupported MPS op. Gated on DEMO_MODE so the
# deployed HF Space (CPU-only, DEMO_MODE=true) is unaffected. Must be set
# before any transitive torch import below.
if os.getenv("DEMO_MODE", "false").lower() != "true":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import streamlit as st

from verirag.config import settings
from verirag.generation.generator import AnswerFormat
from verirag.graph.build import run_query
from verirag.ingest import ingest_file
from verirag.observability.metrics import summarize_metrics
from verirag.retrieval.vector_store import list_ingested_documents

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("./data/raw/uploads")
SUPPORTED_TYPES = ["pdf", "docx", "pptx", "txt", "md"]
QUERY_KEY = "query_input"

FORMAT_LABELS: dict[str, AnswerFormat] = {
    "Concise": AnswerFormat.CONCISE,
    "Detailed": AnswerFormat.DETAILED,
}

# Generic enough to work against any research-paper corpus; the first one is
# built from whatever is actually ingested.
STATIC_EXAMPLES = [
    "What datasets are used for evaluation?",
    "What are the main contributions?",
]

# Demo mode gates the public HF Space: fixed pre-ingested corpus, uploads
# disabled, and a per-session query cap. Local runs leave it off.
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
MAX_QUERIES_PER_SESSION = int(os.getenv("MAX_QUERIES_PER_SESSION", "15"))

st.set_page_config(page_title="VeriRAG", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------
# ingestion / corpus
# --------------------------------------------------------------------------
def _save_upload(uploaded_file) -> Path:
    """Write an uploaded file to UPLOAD_DIR, avoiding filename collisions."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    data = uploaded_file.getbuffer()

    target = UPLOAD_DIR / uploaded_file.name
    if target.exists():
        digest = hashlib.sha256(bytes(data)).hexdigest()[:8]
        target = target.with_name(f"{target.stem}-{digest}{target.suffix}")

    target.write_bytes(data)
    return target


def _refresh_corpus() -> None:
    try:
        # DEMO_MODE uses a fixed pre-ingested corpus shared across sessions.
        # For multi-tenant production, each session would need its own collection.
        st.session_state["corpus"] = list_ingested_documents()
    except Exception as exc:  # noqa: BLE001 - UI must survive a store error
        logger.warning("Could not list corpus: %s", exc)
        st.session_state["corpus"] = []


def _render_system_stats() -> None:
    """Aggregate metrics across every query logged so far."""
    with st.sidebar.expander("System stats"):
        try:
            summary = summarize_metrics()
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Metrics unavailable: {exc}")
            return

        if not summary.total_queries:
            st.caption(
                "No queries logged yet. Ask something and these fill in."
            )
            return

        st.caption(f"Across {summary.total_queries} logged quer"
                   f"{'y' if summary.total_queries == 1 else 'ies'}")
        st.metric("Abstention rate", f"{summary.abstention_rate:.0%}")
        st.metric("Self-correction triggered", f"{summary.loop_trigger_rate:.0%}")
        st.metric("Avg cost / query", f"${summary.avg_cost_per_query_usd:.4f}")

        if summary.p95_latency_by_node:
            st.caption("p95 latency by stage (ms)")
            slowest = sorted(
                summary.p95_latency_by_node.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
            for node, value in slowest[:6]:
                st.text(f"{node:28} {value:9.0f}")


def _render_sidebar() -> None:
    st.sidebar.header("Corpus")

    if not DEMO_MODE:
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
    else:
        st.sidebar.caption(
            "📎 Uploads disabled in demo. Run locally to ingest your own documents."
        )

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
                f"{icon} **{name}**  \n`{row['doc_id'][:20]}` · {row['format']}"
            )

    if st.sidebar.button("Refresh corpus list"):
        _refresh_corpus()

    _render_system_stats()


# --------------------------------------------------------------------------
# result rendering
# --------------------------------------------------------------------------
def _render_verification_row(result) -> None:
    """Gate badges plus the cost/latency row, from values already computed."""
    gate1 = "Gate 1 coverage ✓" if result.gate1_verdict and \
        result.gate1_verdict.verdict == "sufficient" else "Gate 1 coverage"
    gate2 = "Gate 2 groundedness ✓" if result.gate2_verdict and \
        result.gate2_verdict.overall_grounded else "Gate 2 groundedness"
    # safe: no variable interpolation (gate1/gate2 are fixed string literals)
    st.markdown(
        f"<span style='background:#1b5e20;color:#fff;padding:3px 9px;"
        f"border-radius:11px;font-size:0.82em'>{gate1}</span>"
        f"<span style='color:#888;margin:0 8px'>&rarr;</span>"
        f"<span style='background:#1b5e20;color:#fff;padding:3px 9px;"
        f"border-radius:11px;font-size:0.82em'>{gate2}</span>",
        unsafe_allow_html=True,
    )

    corrections = result.regeneration_count
    plural = "" if corrections == 1 else "s"
    st.caption(
        f"**Verified ✓** · {result.total_latency_ms / 1000:.1f}s · "
        f"{corrections} self-correction{plural} · "
        f"~${result.estimated_cost_usd:.4f}"
    )


def _provenance_label(content_types: list[str]) -> str:
    """How the cited text came to exist: extracted, or model-written."""
    if not content_types:
        return "📄 text"
    labels = []
    if "text" in content_types:
        labels.append("📄 text")
    if "figure_caption" in content_types:
        labels.append("🖼️ figure (AI-generated description)")
    return " + ".join(labels) if labels else "📄 text"


def _page_list(pages) -> str:
    """Pages as a sorted, de-duplicated string.

    Citation.pages follows rerank order, which reads as broken once a citation
    spans many chunks ("page(s) 1, 23, 3, 4"). Sorted for display only -- the
    underlying order is left alone, since it reflects retrieval ranking.
    """
    known = sorted({p for p in pages if p is not None})
    return ", ".join(str(p) for p in known)


def _render_sources(citations) -> None:
    st.subheader("Sources")
    for citation in citations:
        provenance = _provenance_label(citation.content_types)
        if citation.source_type == "external":
            st.markdown(
                f"**[{citation.number}]** 🌐 *external* — "
                f"[{citation.url}]({citation.url}) · {provenance}"
            )
        else:
            pages = _page_list(citation.pages)
            page_text = f" · page(s) {pages}" if pages else ""
            st.markdown(
                f"**[{citation.number}]** 📄 *local* — "
                f"`{citation.source_path}`{page_text} · {provenance}"
            )


def _answer_markdown(query: str, result) -> str:
    """The answer and its citations as a self-contained markdown file."""
    lines = [f"# {query}", "", result.answer or "", "", "## Sources", ""]
    for citation in result.citations:
        where = (
            citation.url
            if citation.source_type == "external"
            else citation.source_path
        )
        pages = _page_list(citation.pages)
        page_text = f", page(s) {pages}" if pages else ""
        lines.append(
            f"- **[{citation.number}]** {where}{page_text} "
            f"({citation.source_type}, {_provenance_label(citation.content_types)})"
        )
    lines += [
        "",
        "---",
        "",
        f"Verified by VeriRAG · {result.regeneration_count} self-correction(s) "
        f"· {result.total_latency_ms / 1000:.1f}s · "
        f"~${result.estimated_cost_usd:.4f}",
    ]
    return "\n".join(lines)


def _render_abstention(result) -> None:
    """Say WHICH kind of abstention this is, in plain English."""
    if result.outcome == "abstained_ungrounded":
        st.warning(
            "**Could not verify the answer.** VeriRAG found relevant material "
            "and drafted an answer, but at least one claim could not be "
            "verified against the sources, so the answer was withheld rather "
            "than shown unverified."
        )
    elif result.outcome == "abstained_insufficient":
        st.warning(
            "**Not enough relevant information.** VeriRAG could not find "
            "material in the corpus that addresses this question, so it "
            "declined rather than answering from general knowledge."
        )
    else:
        st.warning("VeriRAG declined to answer.")

    with st.expander("Why", expanded=True):
        st.write(result.abstention_reason or "No reason recorded.")


def _render_result(query: str, result) -> None:
    if result.abstained:
        _render_abstention(result)
    else:
        st.subheader("Answer")
        st.markdown(result.answer or "")
        _render_verification_row(result)
        if result.citations:
            _render_sources(result.citations)
        st.download_button(
            "Download answer + sources (.md)",
            data=_answer_markdown(query, result),
            file_name="verirag-answer.md",
            mime="text/markdown",
        )

    with st.expander("Show reasoning trace"):
        st.caption(
            f"{len(result.trace)} step(s) · "
            f"{result.regeneration_count} regeneration(s) · "
            f"{result.llm_call_count} LLM call(s)"
        )
        for i, step in enumerate(result.trace, start=1):
            st.text(f"{i:2}. {step}")


# --------------------------------------------------------------------------
# main panel
# --------------------------------------------------------------------------
def _example_questions() -> list[str]:
    """Examples that actually fit whatever is ingested right now."""
    corpus = st.session_state.get("corpus") or []
    examples: list[str] = []
    local = [r for r in corpus if r.get("source_type") != "external"]
    if local:
        name = Path(str(local[0]["source_path"])).name
        examples.append(f"Summarize {name}")
    return examples + STATIC_EXAMPLES


def _render_examples() -> None:
    corpus = st.session_state.get("corpus") or []
    if not corpus:
        st.info("Upload a document in the sidebar to get started.")
        return

    st.caption("Try an example:")
    examples = _example_questions()
    for column, example in zip(st.columns(len(examples)), examples):
        with column:
            if st.button(example, key=f"ex_{example}", use_container_width=True):
                st.session_state[QUERY_KEY] = example
                st.rerun()


def _render_main() -> None:
    st.title("🔎 VeriRAG")
    st.caption(
        "A self-verifying document QA system with a two-gate verification "
        "critic. It answers from your documents with citations, or abstains."
    )

    _render_examples()

    query = st.text_input(
        "Question",
        key=QUERY_KEY,
        placeholder="What does the Q4 revenue chart show?",
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

    query_count = st.session_state.get("query_count", 0)
    limit_reached = DEMO_MODE and query_count >= MAX_QUERIES_PER_SESSION
    if limit_reached:
        st.warning(
            "Demo query limit reached (15/session). Refresh to start a new "
            "session, or run the project locally for unlimited queries."
        )

    if st.button(
        "Ask",
        type="primary",
        disabled=limit_reached or not (query or "").strip(),
    ):
        st.session_state["query_count"] = query_count + 1
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
        st.session_state["last_query"] = query.strip()

    if st.session_state.get("last_result") is not None:
        _render_result(
            st.session_state.get("last_query", ""),
            st.session_state["last_result"],
        )


def main() -> None:
    if DEMO_MODE:
        st.info(
            "🔬 **VeriRAG Demo** · Pre-ingested corpus of 5 academic NLP papers. "
            "This demo runs against a fixed corpus; uploads are disabled. "
            "Source: [github.com/YOUR_GITHUB_HANDLE/veri-rag]"
            "(https://github.com/YOUR_GITHUB_HANDLE/veri-rag)",
            icon=None,
        )
    _render_sidebar()
    _render_main()


main()
