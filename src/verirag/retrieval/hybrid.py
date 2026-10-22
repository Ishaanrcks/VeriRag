"""Hybrid retrieval orchestrator: dense + sparse -> RRF -> cross-encoder.

This is the single entry point the rest of VeriRAG should call for
retrieval. It owns the latency/accuracy trade-off: both retrievers run wide,
RRF fuses their rankings, the candidate set is then *capped* before the
expensive cross-encoder pass, and only the final top-k is returned.
"""

from __future__ import annotations

from verirag.config import settings
from verirag.retrieval.fusion import reciprocal_rank_fusion
from verirag.retrieval.reranker import rerank
from verirag.retrieval.sparse_index import sparse_search
from verirag.retrieval.types import SearchHit
from verirag.retrieval.vector_store import dense_search


def _exclude_external(hits: list[SearchHit]) -> list[SearchHit]:
    """Drop web-fetched hits entirely (corpus-only mode).

    Applied to each retriever's output BEFORE fusion, so external chunks never
    enter the ranking at all. Blocking a new fetch is not sufficient on its
    own: web content ingested during an earlier query persists in the stores
    and would otherwise still be retrievable, which would break the
    privacy/determinism guarantee corpus-only mode exists to provide.
    """
    return [
        hit for hit in hits if hit.metadata.get("source_type") != "external"
    ]


def _filter_by_doc_id(hits: list[SearchHit], doc_id: str) -> list[SearchHit]:
    """Keep only hits belonging to one source document.

    Applied to each retriever's output before fusion, like
    :func:`_exclude_external`, so a doc-scoped question ranks only within that
    document instead of competing against the whole corpus.
    """
    return [hit for hit in hits if hit.metadata.get("doc_id") == doc_id]


def _apply_external_penalty(hits: list[SearchHit]) -> list[SearchHit]:
    """Discount external hits' rerank_score, then re-sort.

    Keeps untrusted web content from silently outranking the user's own
    documents when both are comparably relevant. This only DISCOUNTS -- it
    never excludes -- so external material can still surface and win when it
    is genuinely the only relevant evidence, which is the whole point of the
    fallback. A missing ``source_type`` means local (see external/fetcher.py).
    """
    penalty = settings.external_source_score_penalty
    for hit in hits:
        if (
            hit.metadata.get("source_type") == "external"
            and hit.rerank_score is not None
        ):
            hit.rerank_score *= penalty
    return sorted(hits, key=lambda h: h.rerank_score or 0.0, reverse=True)


def hybrid_retrieve(
    query: str,
    corpus_only: bool = False,
    doc_id_filter: str | None = None,
) -> list[SearchHit]:
    """Retrieve the final top-k chunks for a query via the full pipeline.

    With ``corpus_only=True``, previously-ingested external content is
    excluded from the ranking entirely (see :func:`_exclude_external`).
    With ``doc_id_filter`` set, only that document's chunks are ranked
    (see :func:`_filter_by_doc_id`).
    """
    dense_hits = dense_search(query, settings.retrieval_dense_top_k)
    sparse_hits = sparse_search(query, settings.retrieval_sparse_top_k)

    if corpus_only:
        dense_hits = _exclude_external(dense_hits)
        sparse_hits = _exclude_external(sparse_hits)

    if doc_id_filter:
        dense_hits = _filter_by_doc_id(dense_hits, doc_id_filter)
        sparse_hits = _filter_by_doc_id(sparse_hits, doc_id_filter)

    fused = reciprocal_rank_fusion(
        dense_hits, sparse_hits, settings.retrieval_rrf_k
    )

    # Cap before reranking -- the cross-encoder is the expensive stage.
    capped = fused[: settings.retrieval_rerank_candidate_cap]

    reranked = rerank(query, capped, settings.retrieval_final_top_k)
    return _apply_external_penalty(reranked)
