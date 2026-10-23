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
    """Drop web-fetched hits from retrieval. Applied to EVERY query.

    External content is scoped to the query that fetched it. Web pages pulled
    in to answer one question were persisting in the shared store and then
    surfacing for later, unrelated questions -- a query about "summarize this
    paper" fetched a page about summarizing emails, which then competed for
    relevance on every subsequent query.

    So external chunks never enter the ranking at all. A fresh fetch instead
    reaches generation in-memory for that one run (see
    ``fetch_external_node``), and what remains in the store serves only as a
    URL-keyed dedup cache for ``fetch_external``.
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


def _select_diverse(
    hits: list[SearchHit], top_k: int, max_per_doc: int
) -> list[SearchHit]:
    """Take the final top-k, but stop one document from dominating it.

    Cross-paper questions were failing because five chunks drawn from a
    400-chunk, multi-paper corpus cannot hold facts from two papers at once,
    and near-duplicate copies of one paper made it worse (one observed answer
    returned five citations that were all the same paper). Capping each
    doc_id's contribution spreads the budget across distinct sources.

    ``hits`` must already be sorted best-first. If the cap cannot fill top_k
    (a single-document corpus, say), the remainder is backfilled from the
    held-back hits so this never returns fewer results than before.
    """
    selected: list[SearchHit] = []
    overflow: list[SearchHit] = []
    per_doc: dict[str, int] = {}

    for hit in hits:
        if len(selected) >= top_k:
            break
        doc_id = str(hit.metadata.get("doc_id", ""))
        if per_doc.get(doc_id, 0) < max_per_doc:
            selected.append(hit)
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        else:
            overflow.append(hit)

    if len(selected) < top_k:
        selected.extend(overflow[: top_k - len(selected)])
    return selected


def score_external_hits(
    query: str, hits: list[SearchHit], top_k: int | None = None
) -> list[SearchHit]:
    """Rank freshly fetched external chunks for in-memory use in one query.

    Fresh external material deliberately bypasses retrieval, which always
    excludes persisted external content, so it is scored here with the same
    cross-encoder and the same external penalty the normal path would apply.
    That keeps a legitimate fallback working for the query that fetched it
    without letting the material leak into later, unrelated queries.
    """
    if not hits:
        return []
    limit = top_k if top_k is not None else settings.retrieval_final_top_k
    return _apply_external_penalty(rerank(query, hits, limit))


def hybrid_retrieve(
    query: str,
    corpus_only: bool = False,
    doc_id_filter: str | None = None,
) -> list[SearchHit]:
    """Retrieve the final top-k chunks for a query via the full pipeline.

    Persisted external content is ALWAYS excluded from the ranking, for every
    query (see :func:`_exclude_external`) -- a fresh fetch reaches the current
    query in-memory instead. ``corpus_only`` is retained because it also
    blocks a new fetch upstream in the graph's router, and it makes the
    intent explicit at the call site. With ``doc_id_filter`` set, only that
    document's chunks are ranked (see :func:`_filter_by_doc_id`).
    """
    dense_hits = dense_search(query, settings.retrieval_dense_top_k)
    sparse_hits = sparse_search(query, settings.retrieval_sparse_top_k)

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

    # Score every capped candidate, then choose the final set with a
    # per-document cap rather than simply taking the top-k.
    reranked = _apply_external_penalty(rerank(query, capped, len(capped)))
    return _select_diverse(
        reranked,
        settings.retrieval_final_top_k,
        settings.retrieval_max_chunks_per_doc,
    )
