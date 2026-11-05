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
import ipaddress
import json
import logging
import os
import queue
import re
import socket
import shutil
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

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
# Exposed for the app's allowed_paths / PDF page links. Demo files live under
# DEMO_CORPUS_DIR; Gradio stores uploads under the system temp dir.
DEMO_CORPUS_DIR = _DEMO_CORPUS
UPLOAD_TEMP_DIR = Path(tempfile.gettempdir())
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

    # URL downloads staged by a previous process are orphans: the store they were
    # ingested into was a temp dir that went with that process (_URL_STAGE_PREFIX
    # is defined with the add-by-URL helpers below).
    with contextlib.suppress(Exception):
        for entry in UPLOAD_TEMP_DIR.glob(f"{_URL_STAGE_PREFIX}*"):
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)

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
        # the URL downloads kept for this session's citations go with its chunks
        with contextlib.suppress(Exception):
            shutil.rmtree(_url_stage_dir(session_id, create=False), ignore_errors=True)


# ---------------------------------------------------------------------------
# add-by-URL ingest
# ---------------------------------------------------------------------------
# A pasted URL is untrusted input. Downloads are streamed to a temp file, size-
# and page-capped by the SAME limits as uploads (no separate URL limits), checked
# for PDF-ness, then staged in this session's own temp dir for as long as the
# process lives, so citations can link back to the page they came from (see
# _url_stage_dir). Nothing is ever written into demo_corpus/ or the repo, and
# URLs are only logged with the query string stripped, since it may carry a token.
URL_DOWNLOAD_TIMEOUT = 30       # seconds
URL_MAX_REDIRECTS = 5

# Generic client identifier only — no email address, no personal identifier.
_URL_HEADERS = {"User-Agent": "VeriRAG-Demo/1.0 (+document-ingest)"}
# Cap on the HTML we read when scraping a landing page for a PDF link.
_URL_HTML_MAX_BYTES = 2 * 1024 * 1024
# Per-session staging dirs for URL downloads: <temp>/verirag_url_<session id>.
_URL_STAGE_PREFIX = "verirag_url_"
_URL_STREAM_CHUNK = 64 * 1024

_ARXIV_ABS_RE = re.compile(r"^/abs/(?P<ident>.+?)/?$", re.IGNORECASE)
_ANCHOR_RE = re.compile(r"<a\b[^>]*?href\s*=\s*[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
_META_RE = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_META_ATTR_RE = re.compile(r"(name|property|content)\s*=\s*[\"']([^\"']*)[\"']", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _requests():
    """Import requests lazily so a missing dependency breaks only this feature."""
    try:
        import requests
    except ImportError:  # pragma: no cover - requirements pin it
        raise DemoError("Adding documents by URL is unavailable in this deployment.")
    return requests


# ---------------------------------------------------------------------------
# SSRF guard: a pasted URL must never make the server fetch something internal
# (cloud metadata at 169.254.169.254, localhost services, RFC1918 ranges, ...).
# Defence in depth, because each layer alone has a gap:
#   1. every hop (including each redirect, followed by hand) is resolved and ALL
#      its addresses must be public, on port 80/443 only;
#   2. the connected socket's peer address is checked again, so DNS that changes
#      between the check and the connect (rebinding) cannot slip through.
# Set VERIRAG_ALLOW_PRIVATE_URLS=1 only for local development against a fixture
# server; it is off by default and must stay off on a public deployment.
# ---------------------------------------------------------------------------
ALLOW_PRIVATE_URLS = os.getenv("VERIRAG_ALLOW_PRIVATE_URLS", "").lower() in ("1", "true", "yes")
_ALLOWED_PORTS = {80, 443}
_BLOCKED_MSG = "That address is not allowed. Paste a link to a PDF on a public website."
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def _ip_is_public(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _check_url_allowed(url: str) -> None:
    """Raise DemoError unless ``url`` is http(s), on a normal port, and every
    address its host resolves to is public."""
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise DemoError("Paste an http:// or https:// link to a PDF file.")
    if ALLOW_PRIVATE_URLS:
        return
    try:
        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    except ValueError:
        raise DemoError(_BLOCKED_MSG)
    if port not in _ALLOWED_PORTS:
        raise DemoError(_BLOCKED_MSG)
    try:
        infos = socket.getaddrinfo(parts.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise DemoError("Could not reach that URL.")
    addrs = {info[4][0] for info in infos}
    if not addrs or not all(_ip_is_public(a) for a in addrs):
        logger.warning("blocked non-public url target: %s", _safe_url(url))
        raise DemoError(_BLOCKED_MSG)


_SAFE_ADAPTER = None


def _safe_adapter_class(requests):
    """requests adapter whose direct connections re-check the peer address."""
    global _SAFE_ADAPTER
    if _SAFE_ADAPTER is not None:
        return _SAFE_ADAPTER
    from urllib3.connection import HTTPConnection, HTTPSConnection
    from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

    def _check_peer(sock):
        if ALLOW_PRIVATE_URLS:
            return sock
        try:
            peer = sock.getpeername()[0]
        except OSError:
            peer = ""
        if not _ip_is_public(peer):
            with contextlib.suppress(Exception):
                sock.close()
            raise OSError("blocked non-public address")
        return sock

    class _Conn(HTTPConnection):
        def _new_conn(self):
            return _check_peer(super()._new_conn())

    class _SConn(HTTPSConnection):
        def _new_conn(self):
            return _check_peer(super()._new_conn())

    class _Pool(HTTPConnectionPool):
        ConnectionCls = _Conn

    class _SPool(HTTPSConnectionPool):
        ConnectionCls = _SConn

    class _Adapter(requests.adapters.HTTPAdapter):
        def init_poolmanager(self, *a, **k):
            super().init_poolmanager(*a, **k)
            self.poolmanager.pool_classes_by_scheme = {"http": _Pool, "https": _SPool}

    _SAFE_ADAPTER = _Adapter
    return _SAFE_ADAPTER


def _new_session(requests):
    session = requests.Session()
    session.max_redirects = URL_MAX_REDIRECTS
    adapter = _safe_adapter_class(requests)()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _open_stream(session, url: str, requests):
    """GET ``url`` as a stream, following redirects by hand so that EVERY hop
    passes the SSRF check (requests' own redirect following would not)."""
    current = url
    for _ in range(URL_MAX_REDIRECTS + 1):
        _check_url_allowed(current)
        resp = session.get(current, stream=True, timeout=URL_DOWNLOAD_TIMEOUT,
                           allow_redirects=False, headers=_URL_HEADERS)
        location = resp.headers.get("Location")
        if resp.status_code in _REDIRECT_STATUSES and location:
            resp.close()
            current = urljoin(current, location.strip())
            continue
        return resp
    raise requests.exceptions.TooManyRedirects()


def _safe_url(url: str) -> str:
    """URL with query string and fragment removed — the only form we log."""
    try:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except Exception:  # noqa: BLE001
        return "<unparseable url>"


def _http_status_message(status: int) -> str:
    """User-facing reason for an HTTP error status (no raw exception text)."""
    if status in (404, 410):
        return "That URL was not found (404). Check the link and try again."
    if status in (401, 403):
        return "That site refused access to the document (login or permission required)."
    if status == 429:
        return "That site is rate-limiting requests right now. Try again in a minute."
    if status >= 500:
        return f"That site returned a server error ({status}). Try again later."
    return f"That site returned an error ({status})."


def _normalise_url(raw: str) -> str:
    """Strip, default the scheme to https, and reject anything that cannot be a
    web address (spaces, no dot in the host) with a specific message."""
    u = (raw or "").strip()
    if not u:
        raise DemoError("Paste a URL first.")
    scheme = urlsplit(u).scheme.lower()
    if "://" not in u and scheme and scheme not in ("http", "https") and not u.lower().startswith(("mailto:", "javascript:", "data:", "file:")):
        scheme = ""            # "example.com:8080/x" parses as scheme "example.com"
    if "://" in u and scheme not in ("http", "https"):
        raise DemoError("Paste an http:// or https:// link to a PDF file.")
    if u.lower().startswith(("mailto:", "javascript:", "data:", "file:")):
        raise DemoError("Paste an http:// or https:// link to a PDF file.")
    if "://" not in u:
        u = f"https://{u}"
    parts = urlsplit(u)
    host = (parts.hostname or "")
    if re.search(r"\s", u) or "." not in host or host.startswith(".") or host.endswith("."):
        raise DemoError("That doesn't look like a valid URL. Paste a full link, "
                        "e.g. https://arxiv.org/abs/2310.11511")
    return u


def _looks_like_pdf_href(href: str) -> bool:
    path = urlsplit(href).path.lower()
    return path.endswith(".pdf") or "/pdf/" in path


def _scrape_pdf_link(base_url: str, html_text: str) -> str | None:
    """First PDF URL referenced by an HTML page, preferring links whose href or
    link text mentions "pdf"/"download". Returns an absolute http(s) URL."""
    # 1) <meta> hints first: citation_pdf_url and friends are explicit and exact.
    for tag in _META_RE.findall(html_text):
        attrs = {k.lower(): v for k, v in _META_ATTR_RE.findall(tag)}
        key = attrs.get("name", "") or attrs.get("property", "")
        content = attrs.get("content", "")
        if not content:
            continue
        if "pdf" in key.lower() or _looks_like_pdf_href(content):
            cand = _absolute_http_url(base_url, content)
            if cand:
                return cand
    # 2) <a href> candidates. Link TEXT saying "pdf"/"download" is the strongest
    # signal (landing pages often list supplementary PDFs before the paper), then
    # the href itself, then plain document order.
    by_text: list[str] = []
    by_href: list[str] = []
    rest: list[str] = []
    for href, inner in _ANCHOR_RE.findall(html_text):
        if not _looks_like_pdf_href(href):
            continue
        cand = _absolute_http_url(base_url, href)
        if not cand:
            continue
        text = _TAG_RE.sub(" ", inner).lower()
        if any(w in text for w in ("pdf", "download")):
            by_text.append(cand)
        elif any(w in href.lower() for w in ("pdf", "download")):
            by_href.append(cand)
        else:
            rest.append(cand)
    for bucket in (by_text, by_href, rest):
        if bucket:
            return bucket[0]
    return None


def _absolute_http_url(base_url: str, href: str) -> str | None:
    """Absolutise ``href`` against ``base_url``, rejecting non-http(s) schemes
    (javascript:, data:, file: ...)."""
    try:
        absolute = urljoin(base_url, href.strip())
    except Exception:  # noqa: BLE001
        return None
    parts = urlsplit(absolute)
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    return absolute


def resolve_pdf_url(url: str) -> str:
    """Normalise a pasted URL to a direct PDF URL.

    arXiv abstract pages map to their /pdf/ form; anything already ending in
    ``.pdf`` (or an arXiv /pdf/ URL without the suffix) is returned unchanged;
    any other page is fetched and scraped for the first PDF link it references.
    """
    parts = urlsplit(url)
    host = parts.netloc.lower().split(":")[0]
    path = parts.path

    # arxiv.org/abs/XXXX -> arxiv.org/pdf/XXXX
    if host.endswith("arxiv.org"):
        m = _ARXIV_ABS_RE.match(path)
        if m:
            return urlunsplit((parts.scheme or "https", parts.netloc,
                               f"/pdf/{m.group('ident')}", "", ""))
        # arxiv.org/pdf/XXXX (with or without .pdf) is already direct
        if path.lower().startswith("/pdf/"):
            return url

    if path.lower().endswith(".pdf"):
        return url

    # Otherwise: fetch the page and look for a PDF link in it.
    requests = _requests()
    session = _new_session(requests)
    try:
        with _open_stream(session, url, requests) as resp:
            if resp.status_code >= 400:
                raise DemoError(_http_status_message(resp.status_code))
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "pdf" in ctype:
                return url          # the "page" is itself a PDF
            if "html" not in ctype and "xml" not in ctype and ctype:
                raise DemoError("No PDF found at that URL. "
                                "Paste a direct link to a PDF file.")
            body = resp.raw.read(_URL_HTML_MAX_BYTES, decode_content=True) or b""
            final_url = resp.url or url
    except DemoError:
        raise
    except requests.exceptions.Timeout:
        raise DemoError("That URL took too long to respond. "
                        "Try again or upload the file directly.")
    except requests.exceptions.TooManyRedirects:
        raise DemoError("Could not reach that URL.")
    except requests.exceptions.RequestException:
        raise DemoError("Could not reach that URL.")
    finally:
        with contextlib.suppress(Exception):
            session.close()

    html_text = body.decode("utf-8", errors="replace")
    found = _scrape_pdf_link(final_url, html_text)
    if not found:
        raise DemoError("No PDF found at that URL. Paste a direct link to a PDF file.")
    return found


def download_pdf(url: str, max_mb: float) -> Path:
    """Stream a PDF from ``url`` to a temp file, enforcing ``max_mb``.

    The file is a download of untrusted content: it lands in the system temp dir
    only, never in the repo or demo_corpus/."""
    requests = _requests()
    session = _new_session(requests)
    max_bytes = int(max_mb * 1024 * 1024)
    tmp_path: Path | None = None
    try:
        with _open_stream(session, url, requests) as resp:
            if resp.status_code >= 400:
                raise DemoError(_http_status_message(resp.status_code))
            ctype = (resp.headers.get("Content-Type") or "").lower()
            declared_pdf = "pdf" in ctype

            declared_len = resp.headers.get("Content-Length")
            if declared_len and declared_len.isdigit() and int(declared_len) > max_bytes:
                raise DemoError(f"That file is larger than {max_mb:.0f} MB. "
                                "Try a smaller document.")

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
                tmp_path = Path(fh.name)
                total = 0
                first = True
                for chunk in resp.iter_content(chunk_size=_URL_STREAM_CHUNK):
                    if not chunk:
                        continue
                    if first:
                        first = False
                        # Content-Type can lie in both directions; the magic bytes
                        # are the deciding check.
                        if not declared_pdf and not chunk.lstrip()[:5].startswith(b"%PDF-"):
                            raise DemoError("That URL did not return a PDF.")
                    total += len(chunk)
                    if total > max_bytes:
                        raise DemoError(f"That file is larger than {max_mb:.0f} MB. "
                                        "Try a smaller document.")
                    fh.write(chunk)
                if first:          # empty body
                    raise DemoError("That URL did not return a PDF.")
        return tmp_path
    except DemoError:
        _discard_temp(tmp_path)
        raise
    except requests.exceptions.Timeout:
        _discard_temp(tmp_path)
        raise DemoError("That URL took too long to respond. "
                        "Try again or upload the file directly.")
    except requests.exceptions.TooManyRedirects:
        _discard_temp(tmp_path)
        raise DemoError("Could not reach that URL.")
    except requests.exceptions.RequestException:
        _discard_temp(tmp_path)
        raise DemoError("Could not reach that URL.")
    except Exception:
        _discard_temp(tmp_path)
        raise
    finally:
        with contextlib.suppress(Exception):
            session.close()


def _discard_temp(path: Path | None) -> None:
    if path is not None:
        with contextlib.suppress(Exception):
            path.unlink(missing_ok=True)


def _url_stage_dir(session_id: str, create: bool = True) -> Path:
    """This session's directory for documents fetched by URL.

    The download is KEPT for the life of the process rather than deleted after
    ingest: a citation links to the chunk's ``source_path``, so the file has to
    still exist when an answer renders -- exactly as a Gradio upload does. It
    lives under UPLOAD_TEMP_DIR (which app.py hands Gradio as allowed_paths, so
    the page link resolves) and is removed by ``cleanup_session`` and by the
    startup sweep, so nothing a visitor fetched survives a restart."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", session_id or "anon").strip("-._")[:64]
    directory = UPLOAD_TEMP_DIR / f"{_URL_STAGE_PREFIX}{safe or 'anon'}"
    if create:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def _free_staged_path(stage_dir: Path, name: str, size: int) -> Path:
    """A path in ``stage_dir`` for ``name`` that will not clobber a *different*
    document of the same filename (two sites, both "paper.pdf"). Re-fetching the
    same document reuses its path; ingest itself dedupes by content."""
    stem, suffix = name[:-4], name[-4:]          # _display_name_from_url ends .pdf
    candidates = [stage_dir / name]
    candidates += [stage_dir / f"{stem}-{i}{suffix}" for i in range(2, 50)]
    for cand in candidates:
        if not cand.exists() or cand.stat().st_size == size:
            return cand
    return stage_dir / f"{stem}-{uuid.uuid4().hex[:8]}{suffix}"


def _display_name_from_url(url: str) -> str:
    """A readable, sanitised ``*.pdf`` filename derived from the URL path.

    Only ``[A-Za-z0-9._-]`` survives, so a crafted URL cannot escape the staging
    directory or inject path separators."""
    parts = urlsplit(url)
    stem = (parts.path.rstrip("/").rsplit("/", 1)[-1] or "").strip()
    if stem.lower().endswith(".pdf"):
        stem = stem[:-4]
    host = parts.netloc.lower().split(":")[0]
    if not stem and host:
        stem = host
    if "arxiv.org" in host and stem:
        stem = f"arxiv-{stem}"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    cleaned = cleaned[:80] or "document"
    return f"{cleaned}.pdf"


def ingest_from_url(url: str, session_id: str, max_mb: float, max_pages: int) -> tuple[str, bool]:
    """Download the PDF behind ``url`` and ingest it for ``session_id``.

    Goes through the very same guards and ingest entrypoint as an upload, so the
    size/page limits and session tagging are identical. Returns
    ``(display_name, skipped)``; ``skipped`` is True when ingest found the same
    content already in the corpus and added nothing."""
    u = _normalise_url(url)

    pdf_url = resolve_pdf_url(u)
    downloaded = download_pdf(pdf_url, max_mb)

    # Stage under a readable name in this session's directory, so the corpus list
    # and citations show "arxiv-2310.11511.pdf" rather than "tmp8f3k1x.pdf" -- and
    # so the file is still there when an answer cites it (see _url_stage_dir).
    stage_dir = _url_stage_dir(session_id)
    staged = _free_staged_path(stage_dir, _display_name_from_url(pdf_url),
                               downloaded.stat().st_size)
    try:
        downloaded.replace(staged)
    except Exception:
        _discard_temp(downloaded)
        raise
    try:
        validate_upload(str(staged), max_mb, max_pages)
        # ingest_upload takes _REQUEST_LOCK itself; do not wrap it (not reentrant).
        res = ingest_upload(str(staged), session_id)
    except Exception:
        _discard_temp(staged)      # nothing was ingested, so keep no junk around
        raise
    logger.info("ingested from url: %s", _safe_url(pdf_url))
    return staged.name, bool(getattr(res, "skipped", False))


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


def server_key_configured() -> bool:
    """True if a server-side DeepSeek key is available from the environment OR
    .env (pydantic Settings reads both), so the startup warning is accurate."""
    return bool((getattr(settings, "deepseek_api_key", "") or "").strip())


def stream_query(query: str, answer_format, corpus_only: bool, session_id: str,
                 user_key: str | None):
    """Yield friendly step strings as the pipeline runs; the final item is the
    VeriRAGResult wrapped as ('__result__', result) or ('__quota__', own_key)."""
    from verirag.graph.build import run_query_streaming
    from verirag.llm.client import QuotaExceededError

    # The pipeline runs in a worker thread that takes _REQUEST_LOCK for exactly as
    # long as it runs. This generator only relays events, so it never holds the
    # lock while suspended at a yield: if the visitor closes the tab or hits Stop,
    # Gradio abandons/closes this generator, the finally below sets ``cancel``, and
    # the worker stops at the next node boundary and releases the lock.
    events: queue.Queue = queue.Queue()
    cancel = threading.Event()
    done = object()

    def work():
        try:
            with _REQUEST_LOCK:
                if cancel.is_set():
                    return
                with _key_context(user_key):
                    try:
                        gen = run_query_streaming(query, answer_format,
                                                  corpus_only=corpus_only, session_id=session_id)
                        while True:
                            if cancel.is_set():
                                gen.close()
                                return
                            try:
                                _, summary = next(gen)
                                events.put(("step", friendly_step(summary)))
                            except StopIteration as stop:
                                # Missing-facts extraction may make one LLM call;
                                # do it HERE, while the lock and this visitor's key
                                # context are held, so it bills/quotas exactly like
                                # the query itself. The result is cached on the
                                # object, so the UI's later read is free.
                                try:
                                    extract_missing_facts(stop.value, question=query,
                                                          _lock_held=True)
                                except Exception:  # never fail the answer over this extra
                                    logger.exception("missing-facts extraction failed")
                                events.put(("__result__", stop.value))
                                return
                    except QuotaExceededError:
                        events.put(("__quota__", bool(user_key)))
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("query failed")
                        events.put(("__error__", _friendly(exc)))
        finally:
            events.put(done)

    threading.Thread(target=work, name="verirag-query", daemon=True).start()
    try:
        while True:
            item = events.get()
            if item is done:
                return
            yield item
    finally:
        cancel.set()



# ---------------------------------------------------------------------------
# structured missing facts (Feature 2)
# ---------------------------------------------------------------------------
@dataclass
class MissingFacts:
    """What a result could NOT establish, in a form the UI renders as lists and
    later features (recommendations, reports) consume programmatically."""

    unverified_claims: list[str] = field(default_factory=list)  # claims Gate 2 could not ground
    missing_info: list[str] = field(default_factory=list)       # what information was absent
    is_complete: bool = True                                     # True when nothing is missing


_MISSING_CACHE_ATTR = "_verirag_missing_facts"
_MISSING_MAX_ITEMS = 5
_MISSING_ITEM_MAX_CHARS = 120

_MISSING_PROMPT = """A document question-answering system could not fully answer a \
question from the user's documents. Below is what it reported.

Question: {question}

Outcome: {outcome}

System's explanation:
{reason}

Statements it could not ground in the documents:
{claims}

Name the GAPS: the information the QUESTION needs that the user's documents \
do not contain. Rules:
- Name the gap, not the content of a statement. Do NOT repeat or paraphrase \
the unverified statements above; e.g. write "COPD treatment recommendations", \
not "LAMA monotherapy for mild symptoms".
- Merge overlapping gaps into one. Usually 1-3 items; never more than {max_items}.
- Each item is a short noun phrase under 10 words, most important first.
- Do not invent facts or name sources.

Return ONLY a JSON array of strings."""


def _final_claim_rows(result) -> list[dict]:
    """Gate 2's FINAL per-claim verdicts as ``{"text", "verdict", "label"}``.

    Primary source is ``result.pipeline_log``: one ``gate2`` LogEvent per critic
    pass, whose ``detail["claims"]`` carry the bucket (grounded / ungrounded /
    excluded) and the raw label; the LAST such event is the verdict the outcome
    was decided on. Falls back to ``result.gate2_verdict.claims`` (bucketed the
    same way) when the log has no gate2 event."""
    gate2 = [e for e in (getattr(result, "pipeline_log", None) or [])
             if getattr(e, "stage", None) == "gate2"]
    if gate2:
        return list((gate2[-1].detail or {}).get("claims") or [])
    verdict = getattr(result, "gate2_verdict", None)
    if verdict is None:
        return []
    from verirag.pipeline_log import claim_verdict_bucket
    return [{"text": c.claim_text, "verdict": claim_verdict_bucket(c.verdict),
             "label": c.verdict} for c in (getattr(verdict, "claims", None) or [])]


def _gate1_reason(result) -> str | None:
    for e in reversed(getattr(result, "pipeline_log", None) or []):
        if getattr(e, "stage", None) == "gate1":
            return (e.detail or {}).get("llm_reason")
    return None


def _parse_phrase_list(text: str | None) -> list[str]:
    """A JSON array of strings from an LLM reply (tolerates code fences and
    surrounding prose); [] if none can be read."""
    if not text:
        return []
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        items = json.loads(text[start:end + 1])
    except ValueError:
        return []
    out: list[str] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, str):
            continue
        phrase = " ".join(item.split()).strip(" -*.;:")
        if phrase and phrase.lower() not in {p.lower() for p in out}:
            out.append(phrase[:_MISSING_ITEM_MAX_CHARS])
        if len(out) == _MISSING_MAX_ITEMS:
            break
    return out


def _fallback_missing_info(result) -> list[str]:
    """Used only when the LLM call fails on a full abstention, where the UI must
    still say something: the pipeline's own headline, verbatim (not invented)."""
    from verirag.pipeline_log import abstain_parts
    headline, _ = abstain_parts(getattr(result, "abstention_reason", None) or "")
    return [headline[:200]] if headline and headline != "No reason recorded." else []


def _llm_missing_info(result, question: str | None, unverified: list[str]) -> list[str]:
    from verirag.llm.client import chat_completion
    outcome = getattr(result, "outcome", "") or ""
    reason_parts = []
    if getattr(result, "abstention_reason", None):
        reason_parts.append(result.abstention_reason)
    g1 = _gate1_reason(result)
    if g1:
        reason_parts.append(f"Coverage check: {g1}")
    if outcome == "partial" and getattr(result, "answer", None):
        from verirag.generation.partial import split_verified_facts
        body, _ = split_verified_facts(result.answer)
        reason_parts.append(f"Answer as given (partially verified):\n{body[:2000]}")
    prompt = _MISSING_PROMPT.format(
        question=(question or "(not recorded)")[:500],
        outcome=outcome or "unknown",
        reason="\n\n".join(reason_parts)[:4000] or "(none)",
        claims="\n".join(f"- {c}" for c in unverified)[:3000] or "(none)",
        max_items=_MISSING_MAX_ITEMS,
    )
    reply = chat_completion(messages=[{"role": "user", "content": prompt}],
                            temperature=0.0, max_tokens=200)
    return _parse_phrase_list(reply)


def extract_missing_facts(result, question: str | None = None,
                          user_key: str | None = None,
                          _lock_held: bool = False) -> MissingFacts:
    """Structured view of what ``result`` could not establish.

    * unverified_claims: claims in Gate 2's final pass bucketed ``ungrounded``
      (not_supported, external_unsupported, caption_only_uncertified,
      false_hedge). ``excluded`` claims (meta statements, unverifiable hedges)
      are deliberately dropped by the pipeline and are NOT missing facts.
    * missing_info: the pipeline exposes no structured "what was absent" (only
      prose: the abstention reason and Gate 1's LLM reason), so ONE DeepSeek call
      turns that prose into 1-5 noun phrases. Never called for a fully verified
      answer, which gets [] -- nothing is invented.
    * is_complete: no unverified claims AND outcome == "verified".

    Cached on the result object: re-renders and later features read it for free.
    The LLM call runs under ``_REQUEST_LOCK`` + ``_key_context(user_key)`` so it
    uses the visitor's key / the shared budget exactly as the query did;
    ``stream_query`` calls this while it already holds both (``_lock_held``).
    """
    cached = getattr(result, _MISSING_CACHE_ATTR, None)
    if isinstance(cached, MissingFacts):
        return cached

    outcome = getattr(result, "outcome", "verified") or "verified"
    abstained = bool(getattr(result, "abstained", False))

    unverified: list[str] = []
    if outcome != "abstained_insufficient":     # nothing was generated to verify
        seen: set[str] = set()
        for row in _final_claim_rows(result):
            if row.get("verdict") != "ungrounded":
                continue
            text = " ".join(str(row.get("text") or "").split())
            if text and text.lower() not in seen:
                seen.add(text.lower())
                unverified.append(text)

    fully_verified = outcome == "verified" and not abstained
    missing: list[str] = []
    if not fully_verified:
        def call() -> list[str]:
            return _llm_missing_info(result, question, unverified)
        try:
            if _lock_held:
                missing = call()
            else:
                with _REQUEST_LOCK, _key_context(user_key):
                    missing = call()
        except Exception as exc:  # noqa: BLE001 - quota/network: degrade, never fail
            # Includes QuotaExceededError: a verified/partial answer the visitor
            # already has must not be replaced by a quota error over this extra.
            logger.warning("missing-facts call failed (%s): %s", type(exc).__name__, exc)
            missing = []
        if not missing and abstained:
            missing = _fallback_missing_info(result)

    facts = MissingFacts(
        unverified_claims=unverified,
        missing_info=missing,
        is_complete=fully_verified and not unverified,
    )
    with contextlib.suppress(Exception):
        setattr(result, _MISSING_CACHE_ATTR, facts)
    return facts

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
