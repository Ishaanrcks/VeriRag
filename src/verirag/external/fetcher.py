"""External source fetching via Tavily web search.

Gate 1 escalates here when the local corpus does not cover a query. Fetched
results are returned as ``Document`` objects carrying the standard metadata
schema from ``loaders/base.py``, so they flow straight into the existing
ingestion pipeline (chunk -> contextualize -> embed -> store) and become
searchable like any local file.

External Documents additionally carry ``source_type="external"``. Local-file
Documents deliberately do NOT carry the key at all: code that cares treats a
MISSING ``source_type`` as "local", which keeps this change's footprint off
the Phase 1 loaders.

Every failure path degrades to an empty list and logs -- the graph's
fetch_external_node must never be aborted by a web search problem.
"""

from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache

from langchain_core.documents import Document
from tavily import TavilyClient

from verirag.config import settings

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def compute_external_doc_id(url: str) -> str:
    """Stable doc_id for a fetched URL.

    The ``external-`` prefix is deliberately distinct from local doc_ids
    (``<filename_stem>-<hash>``) so external and local material stay visually
    distinguishable even though both live in the same stores.
    """
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"external-{digest}"


def _strip_html(text: str) -> str:
    """Basic markup sanitization: drop tags, collapse whitespace.

    Tavily's content is usually fairly clean already; this is a safety net for
    the spec's content-safety requirement. Tags become spaces rather than being
    deleted outright, so adjacent words are not glued together.
    """
    return _WHITESPACE_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()


def _configured_api_key() -> str | None:
    """Return a usable Tavily key, or None if it is unset or a placeholder.

    ``.env`` files are routinely copied from ``.env.example``, leaving the
    literal ``tvly-...`` placeholder behind. That value is truthy, so a plain
    truthiness check would sail past it and surface as a confusing auth error
    instead of "not configured".
    """
    key = (settings.tavily_api_key or "").strip()
    if not key or "..." in key:
        return None
    return key


@lru_cache(maxsize=1)
def _client(api_key: str) -> TavilyClient:
    return TavilyClient(api_key=api_key)


def fetch_external(query: str) -> list[Document]:
    """Search the web for ``query`` and return it as ingestible Documents."""
    api_key = _configured_api_key()
    if api_key is None:
        logger.warning(
            "TAVILY_API_KEY not configured - external fallback unavailable"
        )
        return []

    try:
        response = _client(api_key).search(
            query=query,
            search_depth=settings.tavily_search_depth,
            max_results=settings.external_max_results,
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never raise
        logger.warning(
            "Tavily search failed (%s): %s", type(exc).__name__, exc
        )
        return []

    results = (response or {}).get("results") or []

    documents: list[Document] = []
    for result in results:
        url = str(result.get("url") or "").strip()
        content = _strip_html(str(result.get("content") or ""))
        if not url or not content:
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "doc_id": compute_external_doc_id(url),
                    "source_path": url,
                    "format": "external_web",
                    "content_type": "text",
                    "page": None,
                    "source_type": "external",
                },
            )
        )

    logger.info(
        "Tavily returned %d usable document(s) of %d result(s) for query=%r",
        len(documents), len(results), query,
    )
    return documents
