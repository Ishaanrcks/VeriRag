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


def hybrid_retrieve(query: str) -> list[SearchHit]:
    """Retrieve the final top-k chunks for a query via the full pipeline."""
    dense_hits = dense_search(query, settings.retrieval_dense_top_k)
    sparse_hits = sparse_search(query, settings.retrieval_sparse_top_k)

    fused = reciprocal_rank_fusion(
        dense_hits, sparse_hits, settings.retrieval_rrf_k
    )

    # Cap before reranking -- the cross-encoder is the expensive stage.
    capped = fused[: settings.retrieval_rerank_candidate_cap]

    return rerank(query, capped, settings.retrieval_final_top_k)
