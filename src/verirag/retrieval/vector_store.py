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
from verirag.embeddings.embedder import embed_query
from verirag.retrieval.types import SearchHit

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


def dense_search(
    query: str, k: int, collection: "chromadb.Collection | None" = None
) -> list[SearchHit]:
    """Dense (vector) search for a query, returning ranked SearchHits.

    The query is embedded with ``embed_query`` -- the *query* side, which
    carries BGE's retrieval instruction prefix (never ``embed_passages``).
    """
    if collection is None:
        collection = get_collection()
    if collection.count() == 0:
        return []

    query_vector = embed_query(query)
    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[SearchHit] = []
    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ),
        start=1,
    ):
        hits.append(
            SearchHit(
                chunk_id=chunk_id,
                raw_text=document or "",
                metadata=dict(metadata or {}),
                dense_rank=rank,
                dense_distance=float(distance),
            )
        )
    return hits


def hydrate_metadata(
    hits: list[SearchHit], collection: "chromadb.Collection | None" = None
) -> list[SearchHit]:
    """Fill in metadata that the BM25 index does not persist.

    The sparse index keeps only a small allowlist of each chunk's metadata,
    and that list has silently dropped a field three times now --
    ``source_type`` (breaking the external penalty and corpus-only filter),
    then ``duplicate_group_id`` and ``parent_text`` (breaking duplicate
    citation merging and small-to-big context). Each time the symptom was
    invisible on the dense path and wrong on the sparse one.

    So rather than extend the allowlist a fourth time, sparse-path hits are
    hydrated here from Chroma, which is the single source of truth for chunk
    metadata, in ONE batched get(). Any field added in future is picked up
    automatically. Values already on the hit are preserved, so a sparse row
    can never be overwritten with something staler.
    """
    if not hits:
        return hits
    if collection is None:
        collection = get_collection()

    result = collection.get(
        ids=[hit.chunk_id for hit in hits], include=["metadatas"]
    )
    stored_by_id = {
        chunk_id: (metadata or {})
        for chunk_id, metadata in zip(
            result.get("ids") or [], result.get("metadatas") or []
        )
    }

    for hit in hits:
        stored = stored_by_id.get(hit.chunk_id)
        if not stored:
            continue
        merged = dict(stored)
        # Keep whatever the hit already carries (never None-out a real value).
        merged.update({k: v for k, v in hit.metadata.items() if v is not None})
        hit.metadata = merged
    return hits


def get_chunks_by_doc_id(
    doc_id: str, collection: "chromadb.Collection | None" = None
) -> list[SearchHit]:
    """Every chunk of one document, in reading order.

    This is a METADATA lookup, not a similarity search: whole-document
    summarization must see all the chunks in order, not the top-k most
    query-similar ones. Ordering uses the ``chunk_index`` the chunker assigns
    at creation time; any chunk missing it sorts last rather than silently
    landing at position 0.
    """
    if collection is None:
        collection = get_collection()

    result = collection.get(
        where={"doc_id": doc_id}, include=["documents", "metadatas"]
    )

    hits: list[SearchHit] = []
    for chunk_id, document, metadata in zip(
        result.get("ids") or [],
        result.get("documents") or [],
        result.get("metadatas") or [],
    ):
        hits.append(
            SearchHit(
                chunk_id=chunk_id,
                raw_text=document or "",
                metadata=dict(metadata or {}),
            )
        )

    def _order(hit: SearchHit) -> int:
        index = hit.metadata.get("chunk_index")
        return index if isinstance(index, int) else 10**9

    hits.sort(key=_order)
    return hits


def list_ingested_documents(
    collection: "chromadb.Collection | None" = None,
) -> list[dict]:
    """One row per distinct source document currently in the store.

    Chroma has no DISTINCT, so metadatas are deduped by doc_id client-side.
    Reads the persistent collection directly, so the UI can show the real
    corpus across restarts.
    """
    if collection is None:
        collection = get_collection()
    if collection.count() == 0:
        return []

    result = collection.get(include=["metadatas"])

    documents: dict[str, dict] = {}
    for metadata in result.get("metadatas") or []:
        if not metadata:
            continue
        doc_id = str(metadata.get("doc_id") or "")
        if not doc_id or doc_id in documents:
            continue
        documents[doc_id] = {
            "doc_id": doc_id,
            "source_path": metadata.get("source_path", "unknown"),
            "format": metadata.get("format", "unknown"),
            # A missing source_type means local (see external/fetcher.py).
            "source_type": metadata.get("source_type", "local"),
            "duplicate_group_id": metadata.get("duplicate_group_id", ""),
        }

    return sorted(documents.values(), key=lambda row: row["source_path"])


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
