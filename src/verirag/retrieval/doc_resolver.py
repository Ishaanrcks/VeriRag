"""Resolve explicit document references in a query.

When a user names an ingested file ("summarize this paper please:
2311.14966v1.pdf"), embedding the whole query -- filename and instruction
included -- searches for *passages that look like a filename*, which nothing
does. Retrieval then returns near-zero scores and Gate 1 correctly abstains.
The gap is that "read this whole document" is a different operation from
"search for content matching this question"; this module detects which one was
asked for.

Matching is deliberately strict -- exact case-insensitive filename, or a
doc_id substring -- with no fuzzy matching, because a false positive silently
narrows retrieval to the wrong document, which is worse than not matching at
all. Note that a bare stem ("2311.14966v1" without ".pdf") does NOT match: a
generic stem like "notes" or "report" would match far too eagerly.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Cheap intent detection. An LLM classifier would be overkill (and would cost
# a call on every query) for a decision this lexically obvious.
_SUMMARIZATION_PATTERNS = [
    r"\bsummari[sz]e\b",
    r"\bsummari[sz]ation\b",
    r"\bsummary\b",
    r"\boverview\b",
    r"\btl;?dr\b",
    r"\bkey (?:points|takeaways)\b",
    r"\bmain (?:points|ideas|findings|takeaways)\b",
    r"\bwhat(?:'s| is) (?:this|the) (?:paper|document|file|pdf) about\b",
]

_SUMMARIZATION_RE = re.compile("|".join(_SUMMARIZATION_PATTERNS), re.IGNORECASE)


def detect_summarization_intent(query: str) -> bool:
    """True if the query reads as "summarize/overview this", not a question."""
    return bool(_SUMMARIZATION_RE.search(query or ""))


def _reference_strings(doc: dict) -> list[str]:
    """The strings that count as naming this document."""
    references: list[str] = []
    source_path = str(doc.get("source_path") or "")
    if source_path:
        filename = Path(source_path).name
        if filename:
            references.append(filename)
    doc_id = str(doc.get("doc_id") or "")
    if doc_id:
        references.append(doc_id)
    return references


def resolve_document_references(
    query: str, ingested_docs: list[dict]
) -> list[str]:
    """Return the doc_ids explicitly named in ``query``.

    ``ingested_docs`` is the output of
    :func:`verirag.retrieval.vector_store.list_ingested_documents`. Usually
    returns 0 or 1 doc_id; a query naming two files for comparison returns
    both, which the graph treats as a multi-document case rather than a
    summarization one.
    """
    lowered_query = (query or "").lower()
    if not lowered_query:
        return []

    matched: list[str] = []
    for doc in ingested_docs or []:
        doc_id = str(doc.get("doc_id") or "")
        if not doc_id or doc_id in matched:
            continue
        if any(ref.lower() in lowered_query for ref in _reference_strings(doc)):
            matched.append(doc_id)

    if matched:
        logger.info(
            "Query names %d ingested document(s): %s", len(matched), matched
        )
    return matched


def strip_document_references(query: str, ingested_docs: list[dict]) -> str:
    """Remove named filenames/doc_ids from the query text.

    A doc-scoped *content* question ("what does 2311.14966v1.pdf say about
    dataset size?") should be embedded as "what does say about dataset size?"
    rather than letting the bare filename pollute the query vector -- the
    document scoping is already handled by the doc_id filter.
    """
    stripped = query or ""
    for doc in ingested_docs or []:
        for reference in _reference_strings(doc):
            if not reference:
                continue
            stripped = re.sub(re.escape(reference), " ", stripped, flags=re.IGNORECASE)

    # Tidy up the punctuation/whitespace the removal leaves behind, e.g.
    # "summarize this paper please: " -> "summarize this paper please".
    stripped = re.sub(r"\s+", " ", stripped)
    stripped = re.sub(r"\s+([,.;:!?])", r"\1", stripped)
    stripped = stripped.strip(" \t\n:;,.-")
    return stripped or (query or "").strip()
