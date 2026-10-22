"""Phase 3 smoke test: full ingestion into dense + sparse stores.

Runs against tests/fixtures/ using ISOLATED temp paths for the Chroma store
and the BM25 corpus (so the real dev store is never touched). Requires a real
DEEPSEEK_API_KEY in .env (contextualization + figure captioning run for real).

    python scripts/test_phase3_ingestion.py
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

# Redirect the stores to throwaway temp locations BEFORE anything reads them.
from verirag.config import settings  # noqa: E402

_TMP_DIR = Path(tempfile.mkdtemp(prefix="verirag_phase3_"))
settings.chroma_persist_dir = str(_TMP_DIR / "chroma")
settings.bm25_corpus_path = str(_TMP_DIR / "bm25_corpus.jsonl")

from verirag.embeddings.embedder import embed_query  # noqa: E402
from verirag.ingest import ingest_directory, ingest_file  # noqa: E402
from verirag.retrieval import vector_store  # noqa: E402
from verirag.retrieval.sparse_index import SparseIndex  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
QUERY = "quarterly revenue chart"


def _rule(title: str) -> None:
    print("=" * 72)
    print(title)


def main() -> None:
    print(f"Isolated stores under: {_TMP_DIR}")

    # 1) Ingest the whole fixtures directory.
    _rule("STEP 1 - ingest_directory(tests/fixtures/)")
    results = ingest_directory(FIXTURES_DIR)
    for r in results:
        print(f"  {r}")
    pdf_result = next((r for r in results if r.doc_id.startswith("sample-")
                       and r.num_figure_chunks > 0), None)

    # 2) Re-ingest sample.txt without force -> should be skipped.
    _rule("STEP 2 - ingest_file(sample.txt) again, no force (expect skipped)")
    again = ingest_file(FIXTURES_DIR / "sample.txt")
    print(f"  {again}")
    print(f"  dedup skip works: {again.skipped is True}")

    # 3) Re-ingest sample.txt WITH force -> clean replace, no duplicates.
    _rule("STEP 3 - ingest_file(sample.txt, force=True) (expect re-ingest)")
    collection = vector_store.get_collection()
    txt_doc_id = again.doc_id
    before_ids = set(collection.get(where={"doc_id": txt_doc_id})["ids"])
    forced = ingest_file(FIXTURES_DIR / "sample.txt", force=True)
    after_ids = set(collection.get(where={"doc_id": txt_doc_id})["ids"])
    print(f"  {forced}")
    print(f"  chunk_ids before={sorted(before_ids)} after={sorted(after_ids)}")
    print(f"  no duplication (id sets equal): {before_ids == after_ids}")

    # 4) Dense vs sparse counts should agree.
    _rule("STEP 4 - store counts")
    dense_count = collection.count()
    sparse_count = SparseIndex().corpus_size
    print(f"  dense (chroma) count : {dense_count}")
    print(f"  sparse (bm25) count  : {sparse_count}")
    print(f"  counts agree: {dense_count == sparse_count}")

    # 5) Raw dense query.
    _rule(f"STEP 5 - raw Chroma query for {QUERY!r} (n_results=3)")
    qvec = embed_query(QUERY)
    res = collection.query(query_embeddings=[qvec.tolist()], n_results=3)
    for cid, doc, dist in zip(
        res["ids"][0], res["documents"][0], res["distances"][0]
    ):
        snippet = doc[:90].replace("\n", " ")
        print(f"  dist={dist:.4f}  chunk_id={cid}")
        print(f"       doc: {snippet!r}")

    # 6) Raw sparse query.
    _rule(f"STEP 6 - sparse BM25 search for {QUERY!r} (k=3)")
    for hit in SparseIndex().search(QUERY, k=3):
        print(f"  score={hit.sparse_score:.4f}  chunk_id={hit.chunk_id}")

    # 7) Delete the PDF's doc_id from both stores; counts drop by its chunks.
    _rule("STEP 7 - delete_by_doc_id(PDF) from both stores")
    if pdf_result is None:
        print("  (no PDF result found; skipping)")
    else:
        pdf_id = pdf_result.doc_id
        pdf_chunks = len(collection.get(where={"doc_id": pdf_id})["ids"])
        dense_before, sparse_before = collection.count(), SparseIndex().corpus_size
        vector_store.delete_by_doc_id(pdf_id)
        sparse = SparseIndex()
        sparse.delete_by_doc_id(pdf_id)
        dense_after, sparse_after = collection.count(), sparse.corpus_size
        print(f"  pdf doc_id={pdf_id} had {pdf_chunks} chunks")
        print(f"  dense : {dense_before} -> {dense_after} "
              f"(drop={dense_before - dense_after})")
        print(f"  sparse: {sparse_before} -> {sparse_after} "
              f"(drop={sparse_before - sparse_after})")
        print(f"  both dropped by {pdf_chunks}: "
              f"{dense_before - dense_after == pdf_chunks and sparse_before - sparse_after == pdf_chunks}")

    _rule("Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
