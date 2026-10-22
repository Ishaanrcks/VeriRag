"""Dense vector store: a thin wrapper over a persistent ChromaDB collection.

We supply our own BGE embeddings (see embeddings/embedder.py), so Chroma's
built-in embedding function is disabled (``embedding_function=None``) and every
write passes explicit vectors. The *retrievable document* stored for each chunk
is the uncontextualized ``raw_text`` so later citations quote the real text,
not the contextual prefix.
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
import numpy as np
from langchain_core.documents import Document

from verirag.config import settings

# Chroma metadata only accepts these scalar types (never None).
_SCALARS = (str, int, float, bool)


@lru_cache(maxsize=4)
def _client(persist_dir: str) -> chromadb.ClientAPI:
    """Cache one PersistentClient per directory (clients are heavyweight)."""
    return chromadb.PersistentClient(path=persist_dir)


def get_collection(persist_dir: str | None = None) -> "chromadb.Collection":
    """Return (creating if needed) the VeriRAG chunk collection.

    Defaults to ``settings.chroma_persist_dir`` but accepts an override so
    tests can point at an isolated temp directory.
    """
    persist_dir = persist_dir or settings.chroma_persist_dir
    return _client(persist_dir).get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=None,  # we always pass our own embeddings
        metadata={"hnsw:space": "cosine"},  # BGE vectors are normalized
    )


def _sanitize_metadata(metadata: dict) -> dict:
    """Make a chunk's metadata Chroma-safe.

    Drops ``raw_text`` (stored separately as the document) and any None value
    (e.g. page=None, context_prefix=None), and coerces non-scalar values to str.
    """
    clean: dict = {}
    for key, value in metadata.items():
        if key == "raw_text" or value is None:
            continue
        clean[key] = value if isinstance(value, _SCALARS) else str(value)
    return clean


def add_chunks(
    chunks: list[Document],
    embeddings: np.ndarray,
    collection: "chromadb.Collection | None" = None,
) -> None:
    """Upsert chunks + embeddings into the collection.

    ids=chunk_id, documents=raw_text (uncontextualized), metadatas=the rest of
    each chunk's (sanitized) metadata.
    """
    if not chunks:
        return
    if collection is None:
        collection = get_collection()

    ids = [c.metadata["chunk_id"] for c in chunks]
    documents = [c.metadata.get("raw_text", c.page_content) for c in chunks]
    metadatas = [_sanitize_metadata(c.metadata) for c in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )


def exists_doc_id(
    doc_id: str, collection: "chromadb.Collection | None" = None
) -> bool:
    """True if any chunk for this source file is already stored (ingest dedup)."""
    if collection is None:
        collection = get_collection()
    result = collection.get(where={"doc_id": doc_id}, limit=1)
    return len(result["ids"]) > 0


def delete_by_doc_id(
    doc_id: str, collection: "chromadb.Collection | None" = None
) -> None:
    """Delete every chunk belonging to a source file (deletion cascade)."""
    if collection is None:
        collection = get_collection()
    collection.delete(where={"doc_id": doc_id})
