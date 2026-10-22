"""Reciprocal Rank Fusion (RRF) of the dense and sparse result lists.

RRF combines rankings without needing the two scorers' scales to be
comparable: each list contributes ``1 / (k + rank)`` for a chunk's 1-based
position in *that* list. A chunk found by both retrievers sums both
contributions, so agreement between the two signals is rewarded, while a
chunk found by only one keeps its single contribution.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

from verirag.retrieval.types import SearchHit


def reciprocal_rank_fusion(
    dense_hits: list[SearchHit],
    sparse_hits: list[SearchHit],
    k: int = 60,
) -> list[SearchHit]:
    """Fuse dense and sparse hits into one list sorted by fused_score desc.

    Input hits are not mutated; merged copies are returned. Per-stage fields
    (dense_rank/dense_distance, sparse_rank/sparse_score) are carried forward
    from whichever list(s) the chunk appeared in, and stay None otherwise.
    """
    merged: "OrderedDict[str, SearchHit]" = OrderedDict()
    fused_scores: dict[str, float] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        copy = replace(hit, metadata=dict(hit.metadata))
        if copy.dense_rank is None:
            copy.dense_rank = rank
        merged[hit.chunk_id] = copy
        fused_scores[hit.chunk_id] = (
            fused_scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
        )

    for rank, hit in enumerate(sparse_hits, start=1):
        existing = merged.get(hit.chunk_id)
        if existing is None:
            copy = replace(hit, metadata=dict(hit.metadata))
            if copy.sparse_rank is None:
                copy.sparse_rank = rank
            merged[hit.chunk_id] = copy
        else:
            # Seen densely too: carry the sparse signal onto the same hit.
            existing.sparse_rank = (
                hit.sparse_rank if hit.sparse_rank is not None else rank
            )
            existing.sparse_score = hit.sparse_score
            if not existing.raw_text:
                existing.raw_text = hit.raw_text
            if not existing.metadata:
                existing.metadata = dict(hit.metadata)
        fused_scores[hit.chunk_id] = (
            fused_scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
        )

    for chunk_id, hit in merged.items():
        hit.fused_score = fused_scores[chunk_id]

    return sorted(
        merged.values(), key=lambda h: h.fused_score or 0.0, reverse=True
    )
