"""External source fetching -- INTERFACE ONLY (stub until Phase 9).

Gate 1 escalates here when the local corpus does not cover a query. Phase 9
will implement real retrieval (arXiv, Tavily) behind this exact signature; the
LangGraph orchestrator already calls ``fetch_external`` by name, so Phase 9
only needs to fill in this function body without touching the graph.

Returning ``list[Document]`` (rather than raw text) means whatever Phase 9
fetches flows straight into the existing ingestion pipeline: the Documents
must carry the standard metadata schema from ``loaders/base.py`` (doc_id,
source_path, format, content_type, page) so they can be chunked, contextual-
ized, embedded, and stored like any local file.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def fetch_external(query: str) -> list[Document]:
    """Fetch external documents for a query. Not implemented until Phase 9.

    Returns an empty list for now, which the graph treats as "the fetch added
    no new material" -- a correct, non-crashing control-flow outcome.
    """
    logger.warning(
        "external fetch not implemented yet (Phase 9); returning no documents "
        "for query=%r",
        query,
    )
    return []
