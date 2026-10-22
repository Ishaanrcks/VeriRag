"""Cross-encoder reranking of fused candidates (BAAI/bge-reranker-base).

A cross-encoder scores the (query, passage) pair jointly, so it is far more
accurate than the bi-encoder used for dense retrieval -- and far more
expensive, since it runs the model once per candidate instead of reusing a
precomputed vector. That is why it only ever sees a small, already-fused
candidate set.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from verirag.config import settings
from verirag.retrieval.types import SearchHit

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Return the lazily-loaded, process-wide cross-encoder singleton."""
    logger.info("Loading reranker model: %s", settings.reranker_model)
    return CrossEncoder(settings.reranker_model)


def rerank(query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
    """Score (query, raw_text) pairs with the cross-encoder; return top_k.

    NOTE: the CALLER is responsible for capping ``hits`` before calling this
    (see settings.retrieval_rerank_candidate_cap). Capping is deliberately
    not done here so this function stays a pure "score exactly what you were
    given" step -- the orchestrator owns the latency/accuracy trade-off.
    """
    if not hits:
        return []

    pairs = [(query, hit.raw_text) for hit in hits]
    scores = get_reranker().predict(pairs)  # one batched forward pass

    for hit, score in zip(hits, scores):
        hit.rerank_score = float(score)

    ranked = sorted(hits, key=lambda h: h.rerank_score or 0.0, reverse=True)
    return ranked[:top_k]
