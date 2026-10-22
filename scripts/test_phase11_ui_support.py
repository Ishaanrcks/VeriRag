"""Phase 11 smoke test: the functions the Streamlit UI depends on.

The UI itself is not easily driven headlessly, so this exercises the logic
behind it against ISOLATED temp stores:

1. The corpus_only retrieval filter -- external content must be excluded
   from the ranking entirely, not merely penalized.
2. list_ingested_documents() -- one row per source document, deduped despite
   multiple chunks per document.

A separate headless `streamlit run` boot check is done outside this script.

    python scripts/test_phase11_ui_support.py
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

# Redirect every store to throwaway temp locations BEFORE anything reads them.
from verirag.config import settings  # noqa: E402

_TMP_DIR = Path(tempfile.mkdtemp(prefix="verirag_phase11_"))
settings.chroma_persist_dir = str(_TMP_DIR / "chroma")
settings.bm25_corpus_path = str(_TMP_DIR / "bm25_corpus.jsonl")
settings.metrics_db_path = str(_TMP_DIR / "metrics.db")

from langchain_core.documents import Document  # noqa: E402

from verirag.external.fetcher import compute_external_doc_id  # noqa: E402
from verirag.ingest import ingest_documents, ingest_file  # noqa: E402
from verirag.retrieval.hybrid import hybrid_retrieve  # noqa: E402
from verirag.retrieval.vector_store import (  # noqa: E402
    list_ingested_documents,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

EXTERNAL_URL = "https://example.com/synthetic-revenue-page"
EXTERNAL_TEXT = (
    "This synthetic web page discusses quarterly revenue charts and the "
    "bar chart titled Quarterly Revenue 2025 in detail."
)
QUERY = "quarterly revenue chart"


def _rule(title: str) -> None:
    print("=" * 76)
    print(title)
    print("=" * 76)


def _synthetic_external_document() -> Document:
    return Document(
        page_content=EXTERNAL_TEXT,
        metadata={
            "doc_id": compute_external_doc_id(EXTERNAL_URL),
            "source_path": EXTERNAL_URL,
            "format": "external_web",
            "content_type": "text",
            "page": None,
            "source_type": "external",
        },
    )


def main() -> None:
    print(f"Isolated stores under: {_TMP_DIR}")

    _rule("SETUP - ingest fixtures + one synthetic external document")
    fixtures = ["sample.pdf", "sample.txt", "sample.pptx"]
    for name in fixtures:
        result = ingest_file(FIXTURES_DIR / name)
        print(f"  {name}: {result}")

    external_doc = _synthetic_external_document()
    external_doc_id = external_doc.metadata["doc_id"]
    for result in ingest_documents([external_doc]):
        print(f"  synthetic external: {result}")
    print(f"  external doc_id: {external_doc_id}")

    _rule("TEST 1 - corpus_only retrieval filter")

    # Assert on doc_id, NOT on metadata["source_type"]: the metadata field is
    # itself part of what can break (it was being dropped by the sparse index),
    # so checking it would let the bug mark its own homework. doc_id carries
    # the "external-" prefix from compute_external_doc_id and is independent.
    def _external_hits(hits: list) -> list:
        return [h for h in hits if h.chunk_id.startswith("external-")]

    def _show(hits: list) -> None:
        for hit in hits:
            kind = hit.metadata.get("source_type") or "local"
            flag = " <-- EXTERNAL" if hit.chunk_id.startswith("external-") else ""
            print(
                f"    [metadata says: {kind:8}] {hit.rerank_score:.6f} "
                f"{hit.chunk_id}{flag}"
            )

    # NOTE: external exclusion is now the DEFAULT for every query, not just
    # corpus-only ones (see _exclude_external). Persisted external content is
    # scoped to the query that fetched it; a fresh fetch reaches generation
    # in-memory instead. So BOTH modes must exclude it here.
    open_hits = hybrid_retrieve(QUERY, corpus_only=False)
    print(f"  corpus_only=False -> {len(open_hits)} hits")
    _show(open_hits)
    open_external = _external_hits(open_hits)

    filtered_hits = hybrid_retrieve(QUERY, corpus_only=True)
    print(f"  corpus_only=True  -> {len(filtered_hits)} hits")
    _show(filtered_hits)
    filtered_external = _external_hits(filtered_hits)

    print(f"  RESULT external excluded under corpus_only=True  : "
          f"{len(filtered_external) == 0}")
    print(f"  RESULT external excluded under corpus_only=False : "
          f"{len(open_external) == 0} (now the default, not just corpus-only)")
    print(f"  RESULT local hits still returned in corpus-only  : "
          f"{len(filtered_hits) > 0}")
    print(f"  RESULT local hits still returned otherwise       : "
          f"{len(open_hits) > 0}")

    _rule("TEST 2 - list_ingested_documents()")
    rows = list_ingested_documents()
    print(f"  {len(rows)} document row(s) returned:")
    for row in rows:
        print(f"    doc_id={row['doc_id'][:28]:28} format={row['format']:14} "
              f"source_type={row['source_type']:8} "
              f"path={Path(str(row['source_path'])).name}")

    doc_ids = [row["doc_id"] for row in rows]
    formats = {row["format"] for row in rows}
    expected_count = len(fixtures) + 1  # 3 fixtures + 1 external
    external_listed = any(
        row["doc_id"] == external_doc_id and row["source_type"] == "external"
        for row in rows
    )
    print(f"  RESULT one row per document ({expected_count} expected): "
          f"{len(rows) == expected_count}")
    print(f"  RESULT no duplicate doc_ids (despite multi-chunk docs): "
          f"{len(doc_ids) == len(set(doc_ids))}")
    print(f"  RESULT external document listed with source_type=external: "
          f"{external_listed}")
    print(f"  RESULT formats seen: {sorted(formats)}")

    _rule("Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
