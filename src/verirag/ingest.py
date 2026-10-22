"""End-to-end ingestion: load -> chunk -> contextualize -> embed -> store.

Writes each file's chunks into both the dense vector store (Chroma) and the
sparse BM25 index, keeping them in sync. Ingestion is idempotent by ``doc_id``:
an already-ingested file is skipped unless ``force=True``, in which case its
existing chunks are deleted from both stores first so a re-ingest replaces
rather than duplicates.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from verirag.chunking.chunker import chunk_documents
from verirag.chunking.contextualize import add_context_to_chunks
from verirag.config import settings
from verirag.embeddings.embedder import embed_passages
from verirag.loaders import load_document
from verirag.loaders.base import EXTENSION_TO_FORMAT, compute_doc_id
from verirag.retrieval import vector_store
from verirag.retrieval.sparse_index import SparseIndex

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Outcome of ingesting a single file."""

    doc_id: str
    num_text_chunks: int
    num_figure_chunks: int
    skipped: bool


def _build_doc_text(documents: list[Document]) -> str:
    """Concatenate all text Documents' content in page/chunk order."""
    return "\n\n".join(
        d.page_content
        for d in documents
        if d.metadata.get("content_type") == "text"
    )


def _ingest_group(
    doc_id: str, documents: list[Document], force: bool = False
) -> IngestResult:
    """Chunk, contextualize, embed, and store one source document's Documents."""
    # Dedup: skip an already-ingested source unless forced.
    if not force and vector_store.exists_doc_id(doc_id):
        logger.info("doc_id=%s already ingested; skipping (use force=True).", doc_id)
        return IngestResult(doc_id, 0, 0, skipped=True)

    sparse = SparseIndex()

    # Force re-ingest: clear existing chunks from both stores first.
    if force:
        vector_store.delete_by_doc_id(doc_id)
        sparse.delete_by_doc_id(doc_id)

    doc_text = _build_doc_text(documents)

    chunks = chunk_documents(documents)
    chunks = add_context_to_chunks(chunks, doc_text)

    # Attach the parent ("big") text for later small-to-big generation.
    parent_text = doc_text[: settings.parent_text_max_chars]
    for chunk in chunks:
        chunk.metadata["parent_text"] = parent_text

    embeddings = embed_passages([c.page_content for c in chunks])

    vector_store.add_chunks(chunks, embeddings)
    sparse.add_chunks(chunks)

    num_text = sum(1 for c in chunks if c.metadata["content_type"] == "text")
    num_figure = sum(
        1 for c in chunks if c.metadata["content_type"] == "figure_caption"
    )
    logger.info(
        "Ingested doc_id=%s: %d text chunks, %d figure chunks.",
        doc_id, num_text, num_figure,
    )
    return IngestResult(doc_id, num_text, num_figure, skipped=False)


def ingest_file(path: str | Path, force: bool = False) -> IngestResult:
    """Load, chunk, contextualize, embed, and store one file into both stores."""
    documents = load_document(path)
    if not documents:
        logger.warning("No Documents loaded from %s; nothing to ingest.", path)
        return IngestResult(compute_doc_id(path), 0, 0, skipped=False)
    return _ingest_group(documents[0].metadata["doc_id"], documents, force=force)


def ingest_documents(
    documents: list[Document], force: bool = False
) -> list[IngestResult]:
    """Ingest already-loaded Documents (e.g. returned by an external fetch).

    Documents are grouped by doc_id so each source is ingested as one unit,
    exactly as ingest_file does for a local file. The graph's external-fetch
    node uses this so fetched material becomes searchable immediately.
    """
    if not documents:
        return []

    groups: "OrderedDict[str, list[Document]]" = OrderedDict()
    for document in documents:
        doc_id = document.metadata.get("doc_id")
        if not doc_id:
            logger.warning(
                "Skipping fetched Document with no doc_id in metadata: %r",
                document.page_content[:80],
            )
            continue
        groups.setdefault(doc_id, []).append(document)

    return [
        _ingest_group(doc_id, docs, force=force) for doc_id, docs in groups.items()
    ]


def ingest_directory(dir_path: str | Path, force: bool = False) -> list[IngestResult]:
    """Ingest every supported file in a directory; skip unsupported ones."""
    directory = Path(dir_path)
    results: list[IngestResult] = []
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in EXTENSION_TO_FORMAT:
            logger.warning("Skipping unsupported file: %s", entry.name)
            continue
        results.append(ingest_file(entry, force=force))
    return results
