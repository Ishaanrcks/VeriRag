"""Hybrid retrieval orchestrator: dense + sparse -> RRF -> cross-encoder.

This is the single entry point the rest of VeriRAG should call for
retrieval. It owns the latency/accuracy trade-off: both retrievers run wide,
RRF fuses their rankings, the candidate set is then *capped* before the
expensive cross-encoder pass, and only the final top-k is returned.
"""

from __future__ import annotations

from verirag.config import SHARED_SESSION_ID, settings
from verirag.retrieval.fusion import reciprocal_rank_fusion
from verirag.retrieval.reranker import rerank
from verirag.retrieval.sparse_index import sparse_search
from verirag.retrieval.types import SearchHit
from verirag.retrieval.vector_store import (
    dense_search,
    hydrate_metadata,
    session_allows,
)


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

    ``hits`` must already be sorted best-first.

    Diversity is spent on the EXTRA slots only. The best-scoring
    ``guaranteed`` hits are taken unconditionally first, because capping from
    the top instead costs single-document questions their most relevant
    chunks: measured, a per-document cap applied from rank 1 dropped a true
    claim from supported to 0.23 by excluding the chunk that stated it. The
    remaining slots then prefer documents not already represented, so
    cross-paper questions gain coverage without single-paper questions losing
    any. Leftovers backfill, so this never returns fewer hits than before.
    """
    guaranteed = min(settings.retrieval_guaranteed_top_k, top_k)
    selected: list[SearchHit] = list(hits[:guaranteed])
    per_doc: dict[str, int] = {}
    for hit in selected:
        key = str(hit.metadata.get("doc_id", ""))
        per_doc[key] = per_doc.get(key, 0) + 1

    remaining = [hit for hit in hits[guaranteed:]]

    # First pass over the extra slots: only documents under the cap.
    held_back: list[SearchHit] = []
    for hit in remaining:
        if len(selected) >= top_k:
            held_back.append(hit)
            continue
        doc_id = str(hit.metadata.get("doc_id", ""))
        if per_doc.get(doc_id, 0) < max_per_doc:
            selected.append(hit)
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        else:
            held_back.append(hit)

    if len(selected) < top_k:
        selected.extend(held_back[: top_k - len(selected)])
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
    session_id: str | None = None,
) -> list[SearchHit]:
    """Retrieve the final top-k chunks for a query via the full pipeline.

    Persisted external content is ALWAYS excluded from the ranking, for every
    query (see :func:`_exclude_external`) -- a fresh fetch reaches the current
    query in-memory instead. ``corpus_only`` is retained because it also
    blocks a new fetch upstream in the graph's router, and it makes the
    intent explicit at the call site. With ``doc_id_filter`` set, only that
    document's chunks are ranked (see :func:`_filter_by_doc_id`).

    ``session_id`` scopes retrieval to shared/demo content plus that session's
    own uploads, applied INSIDE both retrievers (a Chroma ``where`` on the dense
    side, a per-row filter on the sparse side) so the top-k of each is drawn
    from the allowed set, not filtered afterwards.

    ``session_id=None`` applies NO session filter -- the full store, exactly as
    before multi-tenant scoping existed. This keeps trusted callers (the eval
    harness, CLI scripts) working unchanged. The UI always passes a real session
    id, so isolation is enforced for every untrusted/public caller.

    A chunk with a missing/None session_id is treated as SHARED in BOTH paths
    (see ``session_allows``), so legacy content ingested before session tagging
    stays retrievable. The dense side cannot express "missing OR in-set" as a
    Chroma ``where`` (a metadata filter never matches a document lacking the
    field), so it over-fetches and filters in Python to match the sparse side.
    """
    if session_id:
        allowed_sessions = {SHARED_SESSION_ID, session_id}
        # Over-fetch, then null-safe filter in Python (see docstring). The 2x
        # buffer absorbs other-session chunks ranked above this session's own;
        # ample for a demo's concurrency.
        dense_hits = dense_search(query, settings.retrieval_dense_top_k * 2)
        dense_hits = [
            h for h in dense_hits if session_allows(h.metadata, session_id)
        ][: settings.retrieval_dense_top_k]
    else:
        allowed_sessions = None
        dense_hits = dense_search(query, settings.retrieval_dense_top_k)

    # Hydrate sparse hits from Chroma: the BM25 index persists only a subset
    # of chunk metadata, and downstream logic (external penalty, corpus-only
    # filter, duplicate citation merging, small-to-big parent text) all read
    # fields it does not keep.
    sparse_hits = hydrate_metadata(
        sparse_search(
            query,
            settings.retrieval_sparse_top_k,
            allowed_sessions=allowed_sessions,
        )
    )

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
