"""Regression test: sparse-path metadata must match dense-path metadata.

The BM25 index persists only an allowlist of chunk metadata, and that list has
silently dropped a field three times -- source_type, then duplicate_group_id
and parent_text. Each time, the dense path looked fine and the sparse path was
quietly wrong, so a test naming specific keys would just have been updated
after each incident.

This test therefore compares the FULL key sets rather than a named list, so a
field added in future that the sparse index does not persist is caught without
anyone remembering to extend the test.

    python scripts/test_sparse_metadata_parity.py
"""

from __future__ import annotations

import logging
from pathlib import Path

EVAL_STORE = Path("eval_store/queensland")

from verirag.config import settings  # noqa: E402

settings.chroma_persist_dir = str(EVAL_STORE / "chroma")
settings.bm25_corpus_path = str(EVAL_STORE / "bm25_corpus.jsonl")

from verirag.retrieval.hybrid import hybrid_retrieve  # noqa: E402
from verirag.retrieval.sparse_index import sparse_search  # noqa: E402
from verirag.retrieval.vector_store import (  # noqa: E402
    dense_search,
    get_collection,
    hydrate_metadata,
)

logging.basicConfig(level=logging.ERROR)

QUERIES = [
    "graph foundation model parameters",
    "uncertainty quantification prediction intervals",
    "Markov logic networks variational inference",
]


def _chroma_metadata(chunk_ids: list[str]) -> dict[str, dict]:
    result = get_collection().get(ids=chunk_ids, include=["metadatas"])
    return {
        cid: (md or {})
        for cid, md in zip(result.get("ids") or [], result.get("metadatas") or [])
    }


def main() -> None:
    print(f"store: {EVAL_STORE}  chunks: {get_collection().count()}")
    failures: list[str] = []

    print("\n--- raw sparse_search (expected to be INCOMPLETE by design) ---")
    raw_missing: set[str] = set()
    for query in QUERIES:
        for hit in sparse_search(query, 3):
            truth = _chroma_metadata([hit.chunk_id]).get(hit.chunk_id, {})
            raw_missing |= set(truth) - set(hit.metadata)
    print(f"  keys the BM25 rows omit: {sorted(raw_missing) or '(none)'}")

    print("\n--- hydrate_metadata() parity vs Chroma ---")
    for query in QUERIES:
        hits = hydrate_metadata(sparse_search(query, 5))
        truth = _chroma_metadata([h.chunk_id for h in hits])
        for hit in hits:
            expected = truth.get(hit.chunk_id)
            if expected is None:
                continue
            missing = set(expected) - set(hit.metadata)
            mismatched = {
                k for k in set(expected) & set(hit.metadata)
                if hit.metadata[k] != expected[k]
            }
            if missing or mismatched:
                failures.append(
                    f"{hit.chunk_id}: missing={sorted(missing)} "
                    f"mismatched={sorted(mismatched)}"
                )
        print(f"  {query[:44]:44} {len(hits)} hit(s) checked")

    print("\n--- hybrid_retrieve() end-to-end parity ---")
    for query in QUERIES:
        hits = hybrid_retrieve(query, corpus_only=True)
        truth = _chroma_metadata([h.chunk_id for h in hits])
        for hit in hits:
            expected = truth.get(hit.chunk_id)
            if expected is None:
                continue
            missing = set(expected) - set(hit.metadata)
            if missing:
                failures.append(f"hybrid {hit.chunk_id}: missing={sorted(missing)}")
        print(f"  {query[:44]:44} {len(hits)} hit(s) checked")

    print("\n--- the three fields that previously regressed ---")
    for query in QUERIES[:1]:
        for hit in hybrid_retrieve(query, corpus_only=True)[:3]:
            have = [
                k for k in ("source_type", "duplicate_group_id", "parent_text")
                if hit.metadata.get(k) not in (None, "")
            ]
            print(f"  {hit.chunk_id[:40]:40} present: {have}")

    print()
    if failures:
        print(f"FAIL: {len(failures)} parity problem(s)")
        for f in failures[:10]:
            print(f"  {f}")
    else:
        print("PASS: every sparse/hybrid hit's metadata matches Chroma exactly")


if __name__ == "__main__":
    main()
