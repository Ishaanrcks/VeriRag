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
import html
import logging
import os
import re
import uuid
from pathlib import Path

# Local-dev-only: let torch fall back to CPU for ops MPS does not implement,
# instead of raising on an unsupported MPS op. Gated on DEMO_MODE so the
# deployed HF Space (CPU-only, DEMO_MODE=true) is unaffected. Must be set
# before any transitive torch import below.
if os.getenv("DEMO_MODE", "false").lower() != "true":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import streamlit as st

from verirag.config import SHARED_SESSION_ID, settings
from verirag.generation.generator import AnswerFormat
from verirag.graph.build import run_query_streaming
from verirag.ingest import ingest_file
from verirag.llm.client import (
    QuotaExceededError,
    set_session_api_key,
    set_surface_quota,
)
from verirag.loaders.pdf import IMAGES_DIR
from verirag.observability.metrics import summarize_metrics
from verirag.retrieval.vector_store import (
    cleanup_stale_sessions,
    get_collection,
    list_ingested_documents,
)

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

# Bundled demo documents (files in demo_corpus/, licensed in
# demo_corpus/LICENSES.md), shown in the "Sample documents & questions" panel so
# a new visitor has something to try without uploading anything. Each set spans
# capabilities: a plain factual lookup, a question whose answer lives in a table
# or figure, and one the document does NOT answer (to show VeriRAG abstain).
# NOTE: these become answerable once demo_corpus/ is ingested into the shared
# collection (scripts/ingest_demo.py); the research-paper examples above remain.
DEMO_DOCS = [
    {
        "name": "Alice's Adventures in Wonderland — literary text (Project Gutenberg)",
        "questions": [
            "What does Alice find on the little glass table after falling down the rabbit-hole?",
            "Who does Alice meet at the Mad Tea-Party?",
            "What is the price of the White Rabbit's pocket watch?",
        ],
    },
    {
        "name": "CDC NCHS Data Brief No. 529 — COPD in adults (US gov report; tables & figures)",
        "questions": [
            "What percentage of adults aged 18 and over had COPD, by age group?",
            "How did COPD prevalence differ by sex?",
            "What is the recommended treatment for COPD according to this brief?",
        ],
    },
    {
        "name": "Semantic Versioning 2.0.0 — technical spec (CC BY)",
        "questions": [
            "When should the MAJOR version number be incremented?",
            "What do the three parts of a MAJOR.MINOR.PATCH version number mean?",
            "Which programming languages does the spec recommend?",
        ],
    },
]

# Demo mode gates the public HF Space: shared pre-ingested corpus, per-session
# private uploads, and a free-query cap after which a visitor must supply their
# own DeepSeek key. Local runs leave it off (no cap).
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
# Free backend-key queries per session before the visitor must bring their own
# key. Queries made with a user-supplied key do not count toward this.
FREE_QUERY_LIMIT = int(os.getenv("FREE_QUERY_LIMIT", "5"))

st.set_page_config(page_title="VeriRAG", layout="wide")

# Per-session uploads older than this are swept on app startup.
UPLOAD_MAX_AGE_SECONDS = 24 * 3600


# --------------------------------------------------------------------------
# session isolation
# --------------------------------------------------------------------------
def _session_id() -> str:
    """Stable id for this browser session.

    A user's uploads are tagged with it so they are visible only within this
    session; retrieval scopes to shared/demo content plus this id. Generated
    once per session and kept in session state.
    """
    return st.session_state.setdefault("session_id", str(uuid.uuid4()))


@st.cache_resource
def _run_startup_cleanup() -> bool:
    """Once per app process: drop stale per-session uploads. Never raises.

    ``st.cache_resource`` makes this run a single time for the whole server
    process rather than per session, which is the right granularity for a
    housekeeping sweep.
    """
    try:
        removed = cleanup_stale_sessions(UPLOAD_MAX_AGE_SECONDS)
        if removed:
            logger.info(
                "Startup cleanup removed %d stale per-session chunk(s).", removed
            )
    except Exception as exc:  # noqa: BLE001 - cleanup must never block the app
        logger.warning("Startup session cleanup failed: %s", exc)
    return True


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
        # Scope to shared/demo content plus THIS session's own uploads, so one
        # user never sees another user's uploaded documents.
        st.session_state["corpus"] = list_ingested_documents(
            session_id=_session_id()
        )
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

    # Uploads are PRIVATE to this browser session: they are tagged with the
    # session id and only this session can retrieve them (see _session_id and
    # the session-scoped retrieval filter). The shared demo corpus is read-only
    # and common to everyone.
    uploads = st.sidebar.file_uploader(
        "Upload documents (private to your session)",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="PDF, DOCX, PPTX, TXT or MD. Figures in PDFs get captioned. "
             "Only you can query what you upload; it is cleared automatically.",
    )

    if st.sidebar.button("Ingest uploaded files", disabled=not uploads):
        for uploaded_file in uploads or []:
            try:
                saved_path = _save_upload(uploaded_file)
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    result = ingest_file(saved_path, session_id=_session_id())
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
            icon = "(external)" if row["source_type"] == "external" else "(local)"
            name = Path(str(row["source_path"])).name or row["source_path"]
            scope = (
                "demo" if row.get("session_id") == SHARED_SESSION_ID else "yours"
            )
            st.sidebar.markdown(
                f"{icon} **{name}**  \n`{row['doc_id'][:20]}` · "
                f"{row['format']} · _{scope}_"
            )

    if st.sidebar.button("Refresh corpus list"):
        _refresh_corpus()

    _render_system_stats()


# --------------------------------------------------------------------------
# reasoning-trace phrasing
# --------------------------------------------------------------------------
def friendly_step(technical: str) -> str:
    """Plain-language phrasing of a node's technical trace line.

    Presentation only: parses the stable prefix + numbers out of the string
    each graph node already emits (see graph/nodes.py) and restates it in
    plain language. Never changes any count. Unknown lines fall through
    unchanged so nothing is lost and this cannot raise.
    """
    t = (technical or "").strip()

    if t.startswith("resolve_intent:"):
        return "Understanding your question"

    if t.startswith("retrieve:"):
        match = re.search(r"retrieve:\s*(\d+)\s*hits", t)
        n = match.group(1) if match else "several"
        plural = "" if n == "1" else "s"
        return f"Searching your documents — found {n} relevant passage{plural}"

    if t.startswith("gate1:"):
        if "verdict=sufficient" in t:
            return "Checking if there's enough information to answer — yes"
        if "verdict=insufficient" in t:
            return ("Checking if there's enough information to answer — "
                    "not enough found")
        return "Checking if there's enough information to answer"

    if t.startswith("generate:"):
        match = re.search(r"(\d+)\s*citation", t)
        c = match.group(1) if match else "0"
        plural = "" if c == "1" else "s"
        return f"Drafting an answer ({c} source{plural} cited)"

    if t.startswith("gate2:"):
        grounded = "overall_grounded=True" in t
        match = re.search(r"(\d+)\s*unsupported", t)
        unsupported = int(match.group(1)) if match else 0
        if grounded:
            return ("Verifying every claim against your documents — "
                    "all claims verified")
        if unsupported == 0:
            return "Verifying every claim against your documents"
        plural = "" if unsupported == 1 else "s"
        return (f"Verifying every claim against your documents — found "
                f"{unsupported} unsupported claim{plural}")

    if t.startswith("regenerate"):
        match = re.search(r"attempt\s*(\d+)/(\d+)", t)
        if match:
            return (f"Revising the answer to fix unsupported claims "
                    f"(attempt {match.group(1)} of {match.group(2)})")
        return "Revising the answer"

    if t.startswith("finalize_verified:"):
        return "Done — answer passed both verification gates"

    if t.startswith("abstain_ungrounded:"):
        return ("Could not verify this answer — showing you the raw source "
                "passages instead")

    if t.startswith("abstain_insufficient_coverage:"):
        return "Not enough relevant information in your documents to answer"

    if t.startswith("summarize_document:"):
        return "Summarizing the document"

    if t.startswith("fetch_external:"):
        return "Looking for supporting material beyond your documents"

    return technical


# --------------------------------------------------------------------------
# result rendering
# --------------------------------------------------------------------------
def _render_verification_row(result) -> None:
    """Gate badges plus the cost/latency row, from values already computed."""
    gate1 = "Gate 1 coverage" if result.gate1_verdict and \
        result.gate1_verdict.verdict == "sufficient" else "Gate 1 coverage"
    gate2 = "Gate 2 groundedness" if result.gate2_verdict and \
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
        f"**Verified** · {result.total_latency_ms / 1000:.1f}s · "
        f"{corrections} self-correction{plural} · "
        f"~${result.estimated_cost_usd:.4f}"
    )


def _provenance_label(content_types: list[str]) -> str:
    """How the cited text came to exist: extracted, or model-written."""
    if not content_types:
        return "text"
    labels = []
    if "text" in content_types:
        labels.append("text")
    if "figure_caption" in content_types:
        labels.append("figure (AI-generated description)")
    return " + ".join(labels) if labels else "text"


def _page_list(pages) -> str:
    """Pages as a sorted, de-duplicated string.

    Citation.pages follows rerank order, which reads as broken once a citation
    spans many chunks ("page(s) 1, 23, 3, 4"). Sorted for display only -- the
    underlying order is left alone, since it reflects retrieval ranking.
    """
    known = sorted({p for p in pages if p is not None})
    return ", ".join(str(p) for p in known)


# --------------------------------------------------------------------------
# table + image viewers (presentation only)
# --------------------------------------------------------------------------
# content_types whose chunk payload the viewers render directly.
_VIEWER_CONTENT_TYPES = ("table", "figure_caption")


def _media_by_chunk_id(citations) -> dict[str, dict]:
    """Fetch table/figure chunk payloads behind these citations, by chunk_id.

    Presentation only: the aggregated Citation object carries neither a table's
    markdown nor a figure's saved image path, so the viewers read them straight
    from the store (the same ChromaDB our ingestion wrote). Returns
    ``{chunk_id: {content_type, text, page, image_path}}`` for table/figure
    chunks only. Never raises -- a store error just means no viewers render.
    """
    chunk_ids: list[str] = []
    for citation in citations:
        for chunk_id in getattr(citation, "contributing_chunk_ids", []) or []:
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
    if not chunk_ids:
        return {}

    try:
        result = get_collection().get(
            ids=chunk_ids, include=["documents", "metadatas"]
        )
    except Exception as exc:  # noqa: BLE001 - viewers are optional, never crash
        logger.warning("Citation media lookup failed: %s", exc)
        return {}

    media: dict[str, dict] = {}
    for chunk_id, text, metadata in zip(
        result.get("ids") or [],
        result.get("documents") or [],
        result.get("metadatas") or [],
    ):
        metadata = metadata or {}
        content_type = metadata.get("content_type")
        if content_type in _VIEWER_CONTENT_TYPES:
            media[chunk_id] = {
                "content_type": content_type,
                "text": text or "",
                "page": metadata.get("page"),
                "image_path": metadata.get("image_path"),
            }
    return media


def _image_path_is_safe(image_path: str) -> bool:
    """True only if ``image_path`` resolves to a file under IMAGES_DIR.

    image_path comes from ChromaDB metadata written by our own ingestion, but
    the st.image() call is still guarded so nothing outside the images dir can
    ever be served, regardless of what the metadata says.
    """
    try:
        return Path(image_path).resolve().is_relative_to(IMAGES_DIR.resolve())
    except Exception:  # noqa: BLE001 - a bad path is simply not safe
        return False


def _render_citation_media(citation, media_by_id: dict[str, dict]) -> None:
    """Render any table / figure image among a citation's cited chunks."""
    rendered: set[str] = set()
    for chunk_id in getattr(citation, "contributing_chunk_ids", []) or []:
        if chunk_id in rendered:
            continue
        item = media_by_id.get(chunk_id)
        if item is None:
            continue
        rendered.add(chunk_id)
        page = item.get("page")

        if item["content_type"] == "table":
            label = f"Table — page {page}" if page is not None else "Table"
            st.caption(label)
            # Streamlit renders GFM tables natively as HTML tables.
            st.markdown(item["text"])
        elif item["content_type"] == "figure_caption":
            image_path = item.get("image_path")
            if (
                image_path
                and os.path.exists(image_path)
                and _image_path_is_safe(image_path)
            ):
                caption = (
                    f"Figure — page {page}" if page is not None else "Figure"
                )
                try:
                    st.image(
                        image_path, caption=caption, use_container_width=True
                    )
                    if item["text"]:
                        st.caption(item["text"])
                except Exception as exc:  # noqa: BLE001 - never crash on a bad image
                    logger.warning("Could not render image %s: %s", image_path, exc)
            # No image_path / missing file: degrade to the existing caption-only
            # rendering (the source label already marks it as a figure).


def _render_sources(citations) -> None:
    st.subheader("Sources")
    media_by_id = _media_by_chunk_id(citations)
    for citation in citations:
        provenance = _provenance_label(citation.content_types)
        if citation.source_type == "external":
            st.markdown(
                f"**[{citation.number}]** *external* — "
                f"[{citation.url}]({citation.url}) · {provenance}"
            )
        else:
            pages = _page_list(citation.pages)
            page_text = f" · page(s) {pages}" if pages else ""
            st.markdown(
                f"**[{citation.number}]** *local* — "
                f"`{citation.source_path}`{page_text} · {provenance}"
            )
        _render_citation_media(citation, media_by_id)


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


_MAX_UNVERIFIED_EXCERPTS = 5


def _excerpt_source_label(citation) -> str:
    """Same source/page shape as _render_sources, used above each excerpt."""
    provenance = _provenance_label(citation.content_types)
    if citation.source_type == "external":
        return (
            f"**[{citation.number}]** *external* — "
            f"[{citation.url}]({citation.url}) · {provenance}"
        )
    pages = _page_list(citation.pages)
    page_text = f" · page(s) {pages}" if pages else ""
    return (
        f"**[{citation.number}]** *local* — "
        f"`{citation.source_path}`{page_text} · {provenance}"
    )


def _render_unverified_excerpts(citations) -> None:
    """Verbatim passages from the top retrieved chunks, shown when Gate 2 fails.

    No generation call: the text is the Citation.parent_text already computed
    for the failed attempt. Variable content is html.escape()'d because chunk
    text is document-derived, not a fixed literal.
    """
    if not citations:
        return

    top = citations[:_MAX_UNVERIFIED_EXCERPTS]

    st.subheader("Relevant excerpts from your documents")
    st.caption(
        "These are unedited passages from your documents related to your "
        "question. VeriRAG could not construct a fully verified answer from "
        "them — review them yourself."
    )

    media_by_id = _media_by_chunk_id(top)
    for citation in top:
        st.markdown(_excerpt_source_label(citation))
        # safe: fixed-literal HTML scaffolding; the chunk text IS variable
        # document-derived content and is html.escape()'d before interpolation.
        st.markdown(
            "<div style='border-left:3px solid #b26a00;background:#2a2015;"
            "color:#f3e9d7;padding:10px 14px;border-radius:4px;"
            "white-space:pre-wrap;font-size:0.92em;margin:4px 0 12px 0'>"
            f"{html.escape(citation.parent_text)}"
            "</div>",
            unsafe_allow_html=True,
        )
        # Show any cited table (rendered markdown) or figure image alongside
        # the raw parent-text excerpt.
        _render_citation_media(citation, media_by_id)


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

    # Only on ungrounded abstention: show the raw retrieved chunks the
    # generator saw, so the user has something concrete to inspect rather
    # than nothing. Never shown alongside a verified answer.
    if result.outcome == "abstained_ungrounded":
        _render_unverified_excerpts(
            getattr(result, "retrieved_citations", []) or []
        )


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
            # Friendly phrasing as the primary line; exact technical string
            # kept underneath (muted) for a reader who wants the numbers.
            st.markdown(f"{i}. {friendly_step(step)}")
            st.caption(step)


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


def _render_quota_error(using_own_key: bool) -> None:
    """Infrastructure error (DeepSeek quota), styled distinctly from abstention.

    An abstention is a verification outcome (st.warning, 'could not verify'); a
    quota error is a system issue (st.error, 'high traffic'), so a visitor never
    reads a 429 as the system doubting the answer.
    """
    if using_own_key:
        st.error(
            "**API limit reached on your key.** Your DeepSeek key hit its rate "
            "limit or quota. Wait a moment and try again, or use a different key."
        )
    else:
        st.error(
            "**The demo is experiencing high traffic** and has hit its API "
            "quota. Please try again in a few minutes, or enter your own DeepSeek "
            "API key to continue immediately."
        )


def _render_demo_docs() -> None:
    """List bundled demo documents + suggested questions for a new visitor.

    Presentation only. Each question is a button that fills the query box. The
    three questions per document intentionally span a factual lookup, a
    table/figure question, and one the document does not answer (abstention).
    """
    with st.expander("New here? Sample documents & questions to try"):
        st.caption(
            "Bundled demo documents (see demo_corpus/). Each set has a factual "
            "lookup, a question answered by a table or figure, and one the "
            "document does NOT answer — so you can watch VeriRAG abstain rather "
            "than guess."
        )
        for doc_index, doc in enumerate(DEMO_DOCS):
            st.markdown(f"**{doc['name']}**")
            for q_index, question in enumerate(doc["questions"]):
                if st.button(
                    question,
                    key=f"demo_{doc_index}_{q_index}",
                    use_container_width=True,
                ):
                    st.session_state[QUERY_KEY] = question
                    st.rerun()


def _render_main() -> None:
    st.title("VeriRAG")
    st.caption(
        "A self-verifying document QA system with a two-gate verification "
        "critic. It answers from your documents with citations, or abstains."
    )

    _render_demo_docs()
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

    # --- free-query gate + optional per-session key (demo only) ---
    free_used = st.session_state.get("free_query_count", 0)
    user_key = st.session_state.get("user_api_key") or None
    gate_active = DEMO_MODE and free_used >= FREE_QUERY_LIMIT and not user_key

    if DEMO_MODE and not user_key:
        st.caption(
            f"Free demo queries remaining: "
            f"{max(0, FREE_QUERY_LIMIT - free_used)} of {FREE_QUERY_LIMIT}"
        )
    if DEMO_MODE and user_key:
        st.caption("Using your own DeepSeek key for this session.")

    if gate_active:
        st.warning(
            f"You've used your {FREE_QUERY_LIMIT} free demo queries. To continue, "
            "enter your own DeepSeek API key below — it is used only for your "
            "session and never stored."
        )
        entered = st.text_input(
            "Your DeepSeek API key", type="password",
            key="user_api_key_input", placeholder="sk-...",
        )
        if (entered or "").strip():
            st.session_state["user_api_key"] = entered.strip()
            st.rerun()

    if st.button(
        "Ask",
        type="primary",
        disabled=gate_active or not (query or "").strip(),
    ):
        result = None
        # Route this query's DeepSeek calls through the user's key when supplied
        # (else the backend key), and surface a 429 as a distinct error rather
        # than a graceful None that would look like an abstention.
        set_session_api_key(user_key)
        set_surface_quota(True)
        try:
            with st.status("Thinking...", expanded=True) as status:
                try:
                    # run_query_streaming yields (step_name, summary) per node
                    # and returns the final VeriRAGResult via StopIteration.
                    gen = run_query_streaming(
                        query.strip(),
                        FORMAT_LABELS[format_label],
                        corpus_only=corpus_only,
                        session_id=_session_id(),
                    )
                    while True:
                        try:
                            _, summary = next(gen)
                            st.write(friendly_step(summary))
                        except StopIteration as stop:
                            result = stop.value
                            break
                    status.update(label="Done", state="complete")
                except QuotaExceededError:
                    status.update(label="Service busy", state="error")
                    _render_quota_error(using_own_key=bool(user_key))
                    return
                except Exception as exc:  # noqa: BLE001 - surface, do not crash
                    logger.exception("Query failed")
                    status.update(label=f"Query failed: {exc}", state="error")
                    return
        finally:
            set_session_api_key(None)
            set_surface_quota(False)
        # Only backend-key queries count toward the free limit; a visitor using
        # their own key is unlimited (their cost).
        if not user_key:
            st.session_state["free_query_count"] = free_used + 1
        st.session_state["last_result"] = result
        st.session_state["last_query"] = query.strip()

    if st.session_state.get("last_result") is not None:
        _render_result(
            st.session_state.get("last_query", ""),
            st.session_state["last_result"],
        )


def main() -> None:
    _session_id()  # ensure a session id exists before anything uses it
    _run_startup_cleanup()  # once-per-process sweep of stale session uploads
    if DEMO_MODE:
        st.info(
            "**VeriRAG Demo** · Shared pre-ingested sample corpus, plus any "
            "documents you upload (private to your session, cleared "
            "automatically). "
            "Source: [github.com/YOUR_GITHUB_HANDLE/veri-rag]"
            "(https://github.com/YOUR_GITHUB_HANDLE/veri-rag)",
            icon=None,
        )
    _render_sidebar()
    _render_main()


main()
