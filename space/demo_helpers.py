"""Pipeline integration for the VeriRAG Hugging Face Space demo.

Deployment wrapper ONLY. Nothing here changes the ``verirag`` package or the
frozen final-c2 config -- it imports the shipping entrypoints and runs them
exactly as they ship (visual verification on). All UI lives in ``app.py``; this
module is the thin, testable layer between Gradio and the pipeline.

Design notes that matter:

* Per-session isolation is the core's own ``configure_store`` + ``session_id``
  scoping. Each request gets a fresh temp Chroma dir and BM25 corpus; the query
  only sees SHARED content plus that session's chunks; everything is deleted in a
  ``finally`` so nothing a visitor uploads survives the request, let alone a
  restart, or is visible to another visitor.

* Requests are SERIALIZED behind one lock. The demo mutates process-global state
  (the active store path, and -- only on the bring-your-own-key path -- the
  DeepSeek key) for the duration of a request, so two requests must not overlap.
  At demo scale one-at-a-time is fine and makes the key handling provably leak-free.

* BYO key: generation and both critic stages honor ``set_session_api_key`` (a
  ContextVar in verirag.llm.client). Figure captioning during ingest does NOT --
  it reads ``settings.deepseek_api_key`` directly -- so on the BYO path we also
  swap that setting in-memory under the lock and restore it in ``finally``. The
  key is never written to disk and never logged.
"""

from __future__ import annotations

import contextlib
import shutil
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

# --- make the verirag package importable from the Space ---------------------
# The Space must contain the repo's ``src/`` (see README). We add it to the path
# rather than require an install step, so the Space boots with just the files.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pymupdf  # noqa: E402  (PyMuPDF; same dependency the loader uses)

# A single lock serializes every request (see module docstring).
_REQUEST_LOCK = threading.Lock()


@dataclass
class DemoResult:
    """What the UI renders for one query."""

    status: str                    # "verified" | "partial" | "abstained" | "refused"
    verified_answer: str
    candidate_answer: str | None    # only if the pipeline exposes it (flag off by default)
    candidate_reason: str | None
    used_backend_key: bool          # True = ran on the server key (counts against budget)


class DemoError(Exception):
    """A visitor-facing error whose message is safe to show verbatim."""


# --- input guards ------------------------------------------------------------
def validate_pdf(path: str, max_mb: float, max_pages: int) -> int:
    """Raise DemoError with a friendly message if the PDF breaks a limit; else
    return its page count."""
    p = Path(path)
    if not p.exists():
        raise DemoError("No file was received. Please upload a PDF and try again.")
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise DemoError(f"That PDF is {size_mb:.1f} MB; the demo limit is {max_mb:.0f} MB.")
    try:
        with pymupdf.open(path) as doc:
            if doc.is_encrypted:
                raise DemoError("That PDF is password-protected; the demo cannot open it.")
            pages = doc.page_count
    except DemoError:
        raise
    except Exception:
        raise DemoError("That file could not be read as a PDF. Please upload a valid PDF.")
    if pages == 0:
        raise DemoError("That PDF has no pages.")
    if pages > max_pages:
        raise DemoError(f"That PDF has {pages} pages; the demo limit is {max_pages}.")
    return pages


def validate_question(question: str, max_len: int) -> str:
    q = (question or "").strip()
    if not q:
        raise DemoError("Please type a question about the uploaded PDF.")
    if len(q) > max_len:
        raise DemoError(f"That question is {len(q)} characters; the demo limit is {max_len}.")
    return q


# --- the pipeline call -------------------------------------------------------
@contextlib.contextmanager
def _session_store(user_key: str | None):
    """Point the whole pipeline at a throwaway store for one request, and -- on the
    BYO path -- at the visitor's key, restoring everything afterwards.

    Everything in here runs under ``_REQUEST_LOCK`` (held by the caller)."""
    from verirag.config import settings
    from verirag.eval.harness import configure_store
    from verirag.llm.client import set_session_api_key
    from verirag.retrieval import vector_store

    prev_chroma = settings.chroma_persist_dir
    prev_bm25 = getattr(settings, "bm25_corpus_path", None)
    prev_metrics = getattr(settings, "metrics_db_path", None)
    prev_key = settings.deepseek_api_key
    tmp = Path(tempfile.mkdtemp(prefix="verirag_demo_"))
    try:
        configure_store(tmp)
        if user_key:
            # generation + critic honor this ContextVar...
            set_session_api_key(user_key)
            # ...and ingest-time figure captioning reads the setting directly, so
            # swap that too, under the lock, in memory only. Restored below.
            settings.deepseek_api_key = user_key
        yield tmp
    finally:
        set_session_api_key(None)
        settings.deepseek_api_key = prev_key
        settings.chroma_persist_dir = prev_chroma
        if prev_bm25 is not None:
            settings.bm25_corpus_path = prev_bm25
        if prev_metrics is not None:
            settings.metrics_db_path = prev_metrics
        # Drop the cached Chroma client bound to the temp dir, then remove the dir
        # and the uploaded file. Nothing a visitor touched survives the request.
        with contextlib.suppress(Exception):
            vector_store._client.cache_clear()
        shutil.rmtree(tmp, ignore_errors=True)


def run_pipeline(pdf_path: str, question: str, *, user_key: str | None) -> DemoResult:
    """Ingest the PDF into a private per-session store and answer the question,
    exactly as the shipping pipeline does. Serialized; always cleans up.

    ``user_key`` None runs on the server (backend) key; a value runs entirely on
    the visitor's own key."""
    from verirag.generation.generator import AnswerFormat
    from verirag.graph.build import run_query
    from verirag.ingest import ingest_file

    session_id = f"demo-{uuid.uuid4().hex[:16]}"
    with _REQUEST_LOCK:
        with _session_store(user_key):
            try:
                result = ingest_file(pdf_path, session_id=session_id)
            except Exception as exc:  # noqa: BLE001
                raise DemoError(_friendly(exc, "while reading your PDF"))
            stored = getattr(result, "num_text_chunks", 0) + getattr(result, "num_figure_chunks", 0)
            if stored == 0 and not getattr(result, "skipped", False):
                raise DemoError("No text or tables could be extracted from that PDF.")
            try:
                r = run_query(question, AnswerFormat.CONCISE,
                              corpus_only=True, session_id=session_id)
            except Exception as exc:  # noqa: BLE001
                raise DemoError(_friendly(exc, "while answering your question"))

    abstained = bool(getattr(r, "abstained", False))
    outcome = getattr(r, "outcome", "verified")
    status = "abstained" if abstained else outcome
    if abstained and getattr(r, "answer", None):
        status = "refused"
    verified = (r.answer or getattr(r, "abstention_reason", None)
                or "The system did not produce a verified answer.")
    return DemoResult(
        status=status,
        verified_answer=verified,
        candidate_answer=getattr(r, "candidate_answer", None),
        candidate_reason=getattr(r, "candidate_reason", None),
        used_backend_key=user_key is None,
    )


def _friendly(exc: Exception, where: str) -> str:
    """A safe, visitor-facing message -- never a stack trace or a key."""
    text = str(exc).lower()
    if any(w in text for w in ("api key", "unauthorized", "401", "authentication")):
        return "The language-model key was rejected. Check the key and try again."
    if any(w in text for w in ("rate", "quota", "429", "insufficient balance", "payment")):
        return "The language model is rate-limited or out of credit right now. Try again shortly."
    if "timeout" in text or "timed out" in text:
        return f"The request timed out {where}. Please try a shorter document or question."
    return f"Something went wrong {where}. Please try again with a different PDF or question."
