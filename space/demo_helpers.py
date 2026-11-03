"""Frontend <-> backend glue for the VeriRAG Gradio Space.

LIGHTWEIGHT ONLY. This module (and app.py) must not import torch, chromadb,
sentence-transformers, pymupdf or the verirag package — all of that now lives in
the Modal backend (modal_backend.py). Here we only validate input, call the
backend over HTTP, and stream its NDJSON response back to the UI.

The backend URL comes from the MODAL_BACKEND_URL env var (set it as a Space
secret/variable). Example:
    https://<your-account>--veri-rag-backend-fastapi-app.modal.run
"""

from __future__ import annotations

import json
import logging
import os
import uuid

import httpx

logger = logging.getLogger("verirag.demo")

BACKEND_URL = os.getenv("MODAL_BACKEND_URL", "").rstrip("/")
SHARED_SESSION_ID = "__shared__"          # mirrors the backend; demo-corpus scope
_INGEST_TIMEOUT = 600.0                   # ingest + figure captioning can be slow
_QUERY_TIMEOUT = 600.0


class DemoError(Exception):
    """A visitor-facing error whose message is safe to show verbatim."""


def _require_backend() -> str:
    if not BACKEND_URL:
        raise DemoError("The backend URL is not configured (set MODAL_BACKEND_URL).")
    return BACKEND_URL


def new_session_id() -> str:
    return f"demo-{uuid.uuid4().hex[:16]}"


# ---------------------------------------------------------------------------
# input guards (frontend-side; the backend enforces its own limits too)
# ---------------------------------------------------------------------------
def validate_upload(path: str, max_mb: float, max_pages: int) -> None:
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise DemoError("No file was received. Please upload a document and try again.")
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise DemoError(f"'{p.name}' is {size_mb:.1f} MB; the demo limit is {max_mb:.0f} MB.")
    if p.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader           # lightweight, pure-python
            pages = len(PdfReader(str(p)).pages)
        except Exception:
            return                                # let the backend judge a tricky PDF
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
# backend calls
# ---------------------------------------------------------------------------
def startup() -> list[str]:
    """Ping the backend; returns [] (health only). The backend lazily preloads the
    demo corpus on its first /health, so this also warms it."""
    try:
        r = httpx.get(f"{_require_backend()}/health", timeout=120.0)
        r.raise_for_status()
        return []
    except DemoError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("backend health check failed: %s", exc)
        return []


def ingest_upload(path: str, session_id: str):
    """POST the file to the backend; returns a small object with chunk counts."""
    from types import SimpleNamespace
    with open(path, "rb") as fh:
        files = {"file": (os.path.basename(path), fh, "application/octet-stream")}
        r = httpx.post(f"{_require_backend()}/ingest", data={"session_id": session_id},
                       files=files, timeout=_INGEST_TIMEOUT)
    if r.status_code != 200:
        raise DemoError((r.json().get("error") if r.headers.get("content-type", "")
                         .startswith("application/json") else r.text)[:300] or "Ingest failed.")
    d = r.json()
    return SimpleNamespace(doc_id=d.get("doc_id", ""), num_text_chunks=d.get("num_text_chunks", 0),
                           num_figure_chunks=d.get("num_figure_chunks", 0),
                           skipped=d.get("skipped", False))


def list_corpus(session_id: str) -> list[dict]:
    try:
        r = httpx.post(f"{_require_backend()}/corpus", json={"session_id": session_id}, timeout=60.0)
        r.raise_for_status()
        return r.json().get("documents", [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("corpus list failed: %s", exc)
        return []


def system_stats() -> dict | str:
    try:
        r = httpx.post(f"{_require_backend()}/stats", json={}, timeout=60.0)
        r.raise_for_status()
        d = r.json()
        return d.get("message") if "message" in d else d
    except Exception as exc:  # noqa: BLE001
        return f"Metrics unavailable: {exc}"


def cleanup_session(session_id: str) -> None:
    try:
        httpx.post(f"{_require_backend()}/cleanup", json={"session_id": session_id}, timeout=60.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cleanup failed: %s", exc)


def stream_query(question: str, answer_format: str, corpus_only: bool, session_id: str,
                 user_key: str | None):
    """Stream the backend's NDJSON. Yields the same tuples the UI already expects:
    ("step", text) | ("__result__", result_dict) | ("__quota__", own_key) | ("__error__", msg)."""
    body = {"question": question, "session_id": session_id, "format": answer_format,
            "corpus_only": corpus_only, "user_key": user_key or ""}
    try:
        with httpx.stream("POST", f"{_require_backend()}/query", json=body,
                          timeout=_QUERY_TIMEOUT) as r:
            if r.status_code != 200:
                yield ("__error__", f"backend returned HTTP {r.status_code}")
                return
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "step" in obj:
                    yield ("step", obj["step"])
                elif "result" in obj:
                    yield ("__result__", obj["result"])
                elif "quota" in obj:
                    yield ("__quota__", bool(obj["quota"]))
                elif "error" in obj:
                    yield ("__error__", obj["error"])
    except DemoError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("query stream failed")
        yield ("__error__", _friendly(exc))


# ---------------------------------------------------------------------------
# pure-python presentation helpers (no verirag)
# ---------------------------------------------------------------------------
def friendly_step(technical: str) -> str:
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
    if "timeout" in text or "timed out" in text or "readtimeout" in text:
        return "The request timed out. Try a shorter document or question."
    if "connect" in text:
        return "Could not reach the backend. It may be cold-starting; try again in a moment."
    return "Something went wrong. Please try again with a different document or question."
