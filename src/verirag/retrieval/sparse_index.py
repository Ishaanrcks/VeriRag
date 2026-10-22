"""Sparse BM25 index with JSON Lines persistence.

rank_bm25 has no native persistence or incremental update, so we persist the
corpus as JSONL and rebuild an in-memory ``BM25Okapi`` whenever it changes.
Tokenization is a simple lowercased ``\\w+`` regex (no NLTK data download).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from verirag.config import settings
from verirag.retrieval.types import SearchHit

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")

# Lightweight metadata persisted per row for downstream filtering/deletion.
_ROW_METADATA_KEYS = ("doc_id", "source_path", "format", "content_type", "page")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SparseIndex:
    """BM25 index backed by a JSONL corpus file, rebuilt in memory on change."""

    def __init__(self, corpus_path: str | Path | None = None) -> None:
        self.corpus_path = Path(corpus_path or settings.bm25_corpus_path)
        self._rows: list[dict] = []
        self._chunk_ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        if self.corpus_path.exists():
            self.load()

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def corpus_size(self) -> int:
        return len(self._rows)

    def load(self) -> None:
        """Read the JSONL corpus file and (re)build the in-memory index."""
        rows: list[dict] = []
        if self.corpus_path.exists():
            with self.corpus_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        self._rows = rows
        self._rebuild()

    def _rebuild(self) -> None:
        self._chunk_ids = [row["chunk_id"] for row in self._rows]
        if self._rows:
            self._bm25 = BM25Okapi([_tokenize(row["text"]) for row in self._rows])
        else:
            self._bm25 = None

    def _write_all(self) -> None:
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with self.corpus_path.open("w", encoding="utf-8") as fh:
            for row in self._rows:
                fh.write(json.dumps(row) + "\n")

    def add_chunks(self, chunks: list[Document]) -> None:
        """Append chunks (raw_text) to the corpus, then rebuild the index."""
        if not chunks:
            return
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with self.corpus_path.open("a", encoding="utf-8") as fh:
            for chunk in chunks:
                row = {
                    "chunk_id": chunk.metadata["chunk_id"],
                    "text": chunk.metadata.get("raw_text", chunk.page_content),
                    "metadata": {
                        key: chunk.metadata.get(key) for key in _ROW_METADATA_KEYS
                    },
                }
                fh.write(json.dumps(row) + "\n")
                self._rows.append(row)
        self._rebuild()

    def search(self, query: str, k: int) -> list[SearchHit]:
        """Return the top-k BM25 hits as fully-populated SearchHits.

        The corpus rows already hold raw_text and metadata in memory, so the
        hits are complete -- no follow-up lookup in the dense store needed.
        """
        if self._bm25 is None or not self._rows:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        top = np.argsort(scores)[::-1][:k]

        hits: list[SearchHit] = []
        for rank, i in enumerate(top, start=1):
            row = self._rows[i]
            hits.append(
                SearchHit(
                    chunk_id=row["chunk_id"],
                    raw_text=row["text"],
                    metadata=dict(row.get("metadata") or {}),
                    sparse_rank=rank,
                    sparse_score=float(scores[i]),
                )
            )
        return hits

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Drop all rows for a source file, rewrite the JSONL, rebuild."""
        before = len(self._rows)
        self._rows = [
            row for row in self._rows if row["metadata"].get("doc_id") != doc_id
        ]
        if len(self._rows) != before:
            self._write_all()
            self._rebuild()


def sparse_search(
    query: str, k: int, corpus_path: str | Path | None = None
) -> list[SearchHit]:
    """Load the persisted corpus and run a BM25 search (mirrors dense_search).

    Builds a fresh SparseIndex so results always reflect the latest
    ingestion. For large corpora this per-query rebuild should be replaced by
    a cached index invalidated on write (future optimization) -- correctness
    over premature caching here, since a stale cache would silently miss
    newly ingested chunks.
    """
    return SparseIndex(corpus_path).search(query, k)
