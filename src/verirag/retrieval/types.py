"""Shared retrieval types.

``SearchHit`` is the single object a chunk travels in through the whole
retrieval pipeline (dense search -> sparse search -> RRF fusion -> rerank).
Each stage fills in only its own fields and leaves the others untouched; a
field stays ``None`` when the hit was never seen by that stage (e.g. a chunk
found only by BM25 has ``dense_rank is None``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    """One retrieved chunk plus whatever each retrieval stage scored it."""

    chunk_id: str
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Filled by dense_search (1-based rank within the dense result list).
    dense_rank: int | None = None
    dense_distance: float | None = None

    # Filled by sparse search (1-based rank within the BM25 result list).
    sparse_rank: int | None = None
    sparse_score: float | None = None

    # Filled by reciprocal_rank_fusion.
    fused_score: float | None = None

    # Filled by the cross-encoder reranker.
    rerank_score: float | None = None
