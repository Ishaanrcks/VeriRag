"""One-time cleanup: remove web-fetched content from the live stores.

Before external content was scoped to the query that fetched it, pages pulled
in to answer one question accumulated in the shared persistent corpus -- junk
like Wikipedia's "Your papers, please" and an email-summarization blog, fetched
because the raw query text was sent to the web search. They are now excluded
from retrieval, but they still clutter the "Your corpus" sidebar in the UI.

This deletes them from BOTH stores (Chroma and the BM25 JSONL), reusing the
existing delete_by_doc_id primitives rather than hand-editing either store.

Dry run by default -- pass --apply to actually delete:

    python scripts/purge_external_cache.py            # show what would go
    python scripts/purge_external_cache.py --apply    # delete it
"""

from __future__ import annotations

import argparse
import logging

from verirag.config import settings
from verirag.retrieval import vector_store
from verirag.retrieval.sparse_index import SparseIndex

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _is_external(row: dict) -> bool:
    """Match either marker, so nothing slips through on a partial record."""
    return (
        row.get("source_type") == "external"
        or row.get("format") == "external_web"
        or str(row.get("doc_id", "")).startswith("external-")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="actually delete (default: dry run)"
    )
    args = parser.parse_args()

    print(f"chroma_persist_dir : {settings.chroma_persist_dir}")
    print(f"bm25_corpus_path   : {settings.bm25_corpus_path}")

    collection = vector_store.get_collection()
    sparse = SparseIndex()

    rows = vector_store.list_ingested_documents(collection)
    external = [row for row in rows if _is_external(row)]
    local = [row for row in rows if not _is_external(row)]

    print()
    print(f"documents in store : {len(rows)}")
    print(f"  local (keep)     : {len(local)}")
    print(f"  external (purge) : {len(external)}")
    print(f"chunks before      : chroma={collection.count()} "
          f"bm25={sparse.corpus_size}")

    if local:
        print("\nKEEPING these local documents:")
        for row in local:
            print(f"  - {row['doc_id']} | {row['format']} | {row['source_path']}")

    if not external:
        print("\nNothing to purge.")
        return

    print("\nWOULD DELETE:" if not args.apply else "\nDELETING:")
    for row in external:
        print(f"  - {row['doc_id']} | {row['format']} | {row['source_path']}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to delete these.")
        return

    for row in external:
        doc_id = row["doc_id"]
        vector_store.delete_by_doc_id(doc_id, collection)
        sparse.delete_by_doc_id(doc_id)
        print(f"  deleted {doc_id}")

    remaining = [
        row
        for row in vector_store.list_ingested_documents(collection)
        if _is_external(row)
    ]
    print()
    print(f"chunks after       : chroma={collection.count()} "
          f"bm25={SparseIndex().corpus_size}")
    print(f"external documents remaining: {len(remaining)}")
    print("Purge complete." if not remaining else "WARNING: some remain.")


if __name__ == "__main__":
    main()
