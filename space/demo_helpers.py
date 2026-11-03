"""Pipeline integration for the VeriRAG Gradio demo (Hugging Face Space).

Deployment wrapper ONLY — nothing here changes the ``verirag`` package or the
frozen final-c2 config. It drives the shipping entrypoints exactly as they ship
(visual verification on, two-gate critic) and keeps all UI in ``app.py``.

Store model (why it is a single shared store, not a temp dir per session):
the pipeline reads ONE Chroma store, so a pre-loaded demo corpus and a per-visitor
index cannot be two separate dirs. Instead there is one process-local store:

  * the demo corpus (demo_corpus/) is ingested once at startup under the SHARED
    session id — read-only, common to everyone;
  * each visitor's uploads are ingested into the SAME store tagged with their own
    session id;
  * retrieval is session-scoped (SHARED + this session), so a visitor only ever
    sees the demo corpus plus their own uploads, never another visitor's;
  * every non-shared session's chunks are swept at startup, so nothing a visitor
    uploaded survives a restart, and ``cleanup_session`` removes a session on demand.

This matches the Streamlit UI's model. "Ephemeral per session" is achieved by
session-scoping + deletion, which is the only thing that works alongside a shared
demo corpus through a single-store pipeline.

Key handling: generation and both critic stages honor the core's
``set_session_api_key`` ContextVar. Ingest-time figure captioning reads
``settings.deepseek_api_key`` directly, so on the bring-your-own-key path that
setting is also swapped in-memory under a lock and restored in ``finally``. The
key is never written to disk or logged.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import tempfile
import threading
import uuid
from pathlib import Path

logger = logging.getLogger("verirag.demo")

# --- make the verirag package importable from the Space ---------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _cand in (_REPO_ROOT / "src", Path.cwd() / "src"):
    if _cand.exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

import pymupdf  # noqa: E402

from verirag.config import SHARED_SESSION_ID, settings  # noqa: E402
from verirag.eval.harness import configure_store  # noqa: E402

# One process-local store for the whole app (see module docstring). A fresh temp
# dir each process start => nothing persists across a restart except what we
# re-ingest (the demo corpus).
_BASE_STORE = Path(tempfile.mkdtemp(prefix="verirag_demo_store_"))
configure_store(_BASE_STORE)

_DEMO_CORPUS = _REPO_ROOT / "demo_corpus"
# Every query serializes behind this lock: it is the window in which the backend
# key may be swapped for a visitor's key, so two must not overlap.
_REQUEST_LOCK = threading.Lock()


class DemoError(Exception):
    """A visitor-facing error whose message is safe to show verbatim."""


# ---------------------------------------------------------------------------
# startup: preload the demo corpus, sweep any stale visitor uploads
# ---------------------------------------------------------------------------
def startup() -> list[str]:
    """Ingest demo_corpus/ under the shared id (idempotent) and drop every
    non-shared session's chunks. Returns the demo document names."""
    from verirag.ingest import ingest_file
    from verirag.retrieval import vector_store

    # fresh slate: remove anything that is not the shared demo corpus
    with contextlib.suppress(Exception):
        col = vector_store.get_collection()
        got = col.get(include=["metadatas"])
        stale = {m.get("session_id") for m in got["metadatas"]
                 if m.get("session_id") not in (None, SHARED_SESSION_ID)}
        for sid in stale:
            vector_store.delete_by_session(sid)

    names: list[str] = []
    if _DEMO_CORPUS.is_dir():
        # Skip the corpus's own metadata files (README.md, LICENSES.md); they are
        # not demo content and should not be answerable.
        _SKIP = {"readme.md", "licenses.md"}
        for entry in sorted(_DEMO_CORPUS.iterdir()):
            if entry.suffix.lower() not in (".pdf", ".txt", ".md", ".docx", ".pptx"):
                continue
            if entry.name.lower() in _SKIP:
                continue
            try:
                ingest_file(entry, session_id=SHARED_SESSION_ID)
                names.append(entry.name)
            except Exception as exc:  # noqa: BLE001 - a bad demo file must not kill boot
                logger.warning("demo preload failed for %s: %s", entry.name, exc)
    return names


def new_session_id() -> str:
    return f"demo-{uuid.uuid4().hex[:16]}"


# ---------------------------------------------------------------------------
# input guards
# ---------------------------------------------------------------------------
def validate_upload(path: str, max_mb: float, max_pages: int) -> None:
    p = Path(path)
    if not p.exists():
        raise DemoError("No file was received. Please upload a document and try again.")
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise DemoError(f"'{p.name}' is {size_mb:.1f} MB; the demo limit is {max_mb:.0f} MB.")
    if p.suffix.lower() == ".pdf":
        try:
            with pymupdf.open(path) as doc:
                if doc.is_encrypted:
                    raise DemoError(f"'{p.name}' is password-protected; the demo cannot open it.")
                pages = doc.page_count
        except DemoError:
            raise
        except Exception:
            raise DemoError(f"'{p.name}' could not be read as a PDF.")
        if pages > max_pages:
            raise DemoError(f"'{p.name}' has {pages} pages; the demo limit is {max_pages}.")


def validate_question(question: str, max_len: int) -> str:
    q = (question or "").strip()
    if not q:
        raise DemoError("Please type a question.")
    if len(q) > max_len:
        raise DemoError(f"That question is {len(q)} characters; the demo limit is {max_len}.")
    return q


# ---------------------------------------------------------------------------
# ingest / corpus listing / stats / cleanup
# ---------------------------------------------------------------------------
def ingest_upload(path: str, session_id: str):
    """Ingest one uploaded file into this visitor's session. Returns IngestResult."""
    from verirag.ingest import ingest_file
    with _REQUEST_LOCK:
        return ingest_file(path, session_id=session_id)


def list_corpus(session_id: str) -> list[dict]:
    """Shared demo docs plus this session's uploads."""
    from verirag.retrieval.vector_store import list_ingested_documents
    try:
        rows = list_ingested_documents() or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("corpus list failed: %s", exc)
        return []
    out = []
    for r in rows:
        sid = r.get("session_id")
        if sid in (None, SHARED_SESSION_ID) or sid == session_id:
            out.append(r)
    return out


def system_stats() -> dict | str:
    """Aggregate metrics, or a message string when unavailable/empty."""
    try:
        from verirag.observability.metrics import summarize_metrics
        s = summarize_metrics()
    except Exception as exc:  # noqa: BLE001
        return f"Metrics unavailable: {exc}"
    if not s.total_queries:
        return "No queries logged yet. Ask something and these fill in."
    return dict(
        total=s.total_queries, abstention_rate=s.abstention_rate,
        loop_trigger_rate=s.loop_trigger_rate, avg_cost=s.avg_cost_per_query_usd,
        p95=sorted(s.p95_latency_by_node.items(), key=lambda kv: kv[1], reverse=True)[:6],
    )


def cleanup_session(session_id: str) -> None:
    from verirag.retrieval import vector_store
    if session_id and session_id != SHARED_SESSION_ID:
        with contextlib.suppress(Exception):
            vector_store.delete_by_session(session_id)


# ---------------------------------------------------------------------------
# streaming query
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _key_context(user_key: str | None):
    from verirag.llm.client import set_session_api_key, set_surface_quota
    prev = settings.deepseek_api_key
    try:
        set_surface_quota(True)
        if user_key:
            set_session_api_key(user_key)
            settings.deepseek_api_key = user_key   # vision captioning reads this directly
        yield
    finally:
        set_session_api_key(None)
        set_surface_quota(False)
        settings.deepseek_api_key = prev


def stream_query(query: str, answer_format, corpus_only: bool, session_id: str,
                 user_key: str | None):
    """Yield friendly step strings as the pipeline runs; the final item is the
    VeriRAGResult wrapped as ('__result__', result) or ('__quota__', own_key)."""
    from verirag.graph.build import run_query_streaming
    from verirag.llm.client import QuotaExceededError

    with _REQUEST_LOCK:
        with _key_context(user_key):
            try:
                gen = run_query_streaming(query, answer_format,
                                          corpus_only=corpus_only, session_id=session_id)
                while True:
                    try:
                        _, summary = next(gen)
                        yield ("step", friendly_step(summary))
                    except StopIteration as stop:
                        yield ("__result__", stop.value)
                        return
            except QuotaExceededError:
                yield ("__quota__", bool(user_key))
            except Exception as exc:  # noqa: BLE001
                logger.exception("query failed")
                yield ("__error__", _friendly(exc))


# ---------------------------------------------------------------------------
# citation media + friendly phrasing (ported from the Streamlit UI)
# ---------------------------------------------------------------------------
_VIEWER_CONTENT_TYPES = {"table", "figure_caption"}


def media_by_chunk_id(citations) -> dict:
    """``{chunk_id: {content_type, text, page, image_path}}`` for table/figure
    chunks behind these citations. Read straight from the store, like the
    Streamlit UI does, because the aggregated Citation carries neither a table's
    markdown nor a figure's image path. Never raises."""
    from verirag.retrieval.vector_store import get_collection
    ids: list[str] = []
    for c in citations:
        for cid in getattr(c, "contributing_chunk_ids", []) or []:
            if cid not in ids:
                ids.append(cid)
    if not ids:
        return {}
    try:
        got = get_collection().get(ids=ids, include=["documents", "metadatas"])
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for cid, text, meta in zip(got.get("ids") or [], got.get("documents") or [],
                               got.get("metadatas") or []):
        if meta.get("content_type") in _VIEWER_CONTENT_TYPES:
            out[cid] = dict(content_type=meta.get("content_type"), text=text,
                            page=meta.get("page"), image_path=meta.get("image_path"))
    return out


def image_path_is_safe(image_path: str) -> bool:
    """Only render images the pipeline itself wrote, under IMAGES_DIR."""
    try:
        from verirag.loaders.pdf import IMAGES_DIR
        return Path(image_path).resolve().is_relative_to(Path(IMAGES_DIR).resolve())
    except Exception:  # noqa: BLE001
        return False


def friendly_step(technical: str) -> str:
    """Plain-language phrasing of a node's technical trace line (ported from
    src/verirag/ui/app.py so the Space has no dependency on the Streamlit module)."""
    t = technical or ""
    low = t.lower()
    if low.startswith("retrieve"):
        return "Searching your documents for relevant passages"
    if low.startswith("gate1") or "coverage" in low:
        return "Checking whether the passages cover the question"
    if low.startswith("generate") or low.startswith("regenerate"):
        return "Drafting an answer from the retrieved passages"
    if low.startswith("gate2") or "grounded" in low:
        return "Verifying every claim against the sources"
    if low.startswith("finalize_partial") or "partial" in low:
        return "Keeping only the verified statements"
    if low.startswith("abstain"):
        return "Declining — the sources do not support an answer"
    if low.startswith("finalize") or low.startswith("answer"):
        return "Finalizing the verified answer"
    if t.startswith("summarize"):
        return "Summarizing the document"
    if t.startswith("fetch_external:"):
        return "Looking for supporting material beyond your documents"
    return t


def _friendly(exc: Exception) -> str:
    text = str(exc).lower()
    if any(w in text for w in ("api key", "unauthorized", "401", "authentication")):
        return "The language-model key was rejected. Check the key and try again."
    if any(w in text for w in ("rate", "quota", "429", "insufficient balance", "payment")):
        return "The language model is rate-limited or out of credit right now. Try again shortly."
    if "timeout" in text or "timed out" in text:
        return "The request timed out. Try a shorter document or question."
    return "Something went wrong. Please try again with a different document or question."
