"""Local sentence-transformers embeddings (BAAI/bge-base-en-v1.5).

IMPORTANT — BGE query/passage asymmetry:
  * PASSAGES (documents/chunks) are embedded with **no** instruction prefix.
  * QUERIES are embedded **with** the retrieval instruction prefix below.

bge-base-en-v1.5 was trained this way: only the query side carries the
"Represent this sentence for searching relevant passages:" instruction.
Getting this backwards — prefixing passages, or dropping it on queries —
silently hurts retrieval recall, so the two code paths are kept separate and
explicit on purpose.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from verirag.config import settings

logger = logging.getLogger(__name__)

# Prepend ONLY to queries (see module docstring).
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Return the lazily-loaded, process-wide embedding model singleton."""
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed passages/chunks (NO instruction prefix). Returns (n, dim) float32."""
    model = get_embedder()
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def embed_query(text: str) -> np.ndarray:
    """Embed a single query (WITH instruction prefix). Returns (dim,) float32.

    Built now for Phase 4 retrieval so the prefix asymmetry lives in one place.
    """
    model = get_embedder()
    return model.encode(
        _QUERY_INSTRUCTION + text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
