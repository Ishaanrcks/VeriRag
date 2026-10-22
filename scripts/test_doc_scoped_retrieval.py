"""Test filename-aware, document-scoped retrieval (and its guardrails).

Covers:
  (b) a content question naming a real filename retrieves only from that doc
  (c) a query naming a file that is NOT in the corpus degrades gracefully to
      normal full-corpus retrieval rather than crashing or returning nothing
  (d) REGRESSION: the cross-document presupposition query must STILL abstain.
      This fix must not make Gate 1 more permissive.
  plus an end-to-end check that a BM25-path hit really carries its doc_id,
  rather than trusting that the field exists in the dataclass.

Uses an ISOLATED temp store but ingests the REAL uploaded papers, so (d)
reproduces the original conditions.

    python scripts/test_doc_scoped_retrieval.py
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

# Redirect every store to throwaway temp locations BEFORE anything reads them.
from verirag.config import settings  # noqa: E402

_TMP_DIR = Path(tempfile.mkdtemp(prefix="verirag_docscope_"))
settings.chroma_persist_dir = str(_TMP_DIR / "chroma")
settings.bm25_corpus_path = str(_TMP_DIR / "bm25_corpus.jsonl")
settings.metrics_db_path = str(_TMP_DIR / "metrics.db")

from verirag.generation.generator import AnswerFormat  # noqa: E402
from verirag.graph.build import run_query  # noqa: E402
from verirag.ingest import ingest_file  # noqa: E402
from verirag.retrieval.sparse_index import sparse_search  # noqa: E402
from verirag.retrieval.vector_store import (  # noqa: E402
    list_ingested_documents,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOADS = PROJECT_ROOT / "data" / "raw" / "uploads"
PAPER_A = UPLOADS / "2311.14966v1.pdf"
PAPER_B = UPLOADS / "Anderson-and-Krathwohl_Revised-Blooms-Taxonomy.pdf"

SCOPED_QUERY = "what does 2311.14966v1.pdf say about dataset size?"
MISSING_FILE_QUERY = "what does nonexistent-paper-12345.pdf say about taxonomy?"
REGRESSION_QUERY = "what is the taxonomy introduced in all of these papers"


def _rule(title: str) -> None:
    print("=" * 76)
    print(title)
    print("=" * 76)


def _report(result) -> None:
    print(f"-- TRACE ({len(result.trace)} steps) --")
    for i, step in enumerate(result.trace, start=1):
        print(f"  {i:2}. {step}")
    print(f"  abstained={result.abstained} "
          f"regenerations={result.regeneration_count}")
    if result.abstained:
        print("  abstention_reason:")
        for line in (result.abstention_reason or "").splitlines():
            print(f"      {line}")
    else:
        answer = (result.answer or "").replace("\n", " ")
        print(f"  answer[:200]: {answer[:200]!r}")
    doc_ids = sorted({c.doc_id for c in result.citations})
    print(f"  citation doc_ids: {doc_ids or '(none)'}")


def main() -> None:
    print(f"Isolated stores under: {_TMP_DIR}")

    missing = [p.name for p in (PAPER_A, PAPER_B) if not p.exists()]
    if missing:
        print(f"SKIP: missing uploads {missing}. Upload them first.")
        return

    _rule("INGEST both real papers (this takes several minutes)")
    doc_ids: dict[str, str] = {}
    for paper in (PAPER_A, PAPER_B):
        result = ingest_file(paper)
        doc_ids[paper.name] = result.doc_id
        print(f"  {paper.name}: {result}")

    doc_a = doc_ids[PAPER_A.name]
    doc_b = doc_ids[PAPER_B.name]
    print(f"  corpus: {[r['doc_id'] for r in list_ingested_documents()]}")

    _rule("STEP 1 CHECK - doc_id survives the BM25 path end to end")
    sparse_hits = sparse_search("taxonomy cognitive process dimension", k=5)
    print(f"  {len(sparse_hits)} sparse hit(s)")
    for hit in sparse_hits:
        print(f"    score={hit.sparse_score:.4f} doc_id="
              f"{hit.metadata.get('doc_id')} chunk={hit.chunk_id}")
    known = set(doc_ids.values())
    all_have_doc_id = all(hit.metadata.get("doc_id") in known for hit in sparse_hits)
    consistent = all(
        hit.chunk_id.startswith(str(hit.metadata.get("doc_id")))
        for hit in sparse_hits
    )
    print(f"  every BM25 hit carries a known doc_id : {all_have_doc_id}")
    print(f"  doc_id consistent with chunk_id prefix: {consistent}")

    _rule(f"TEST (b) - doc-scoped question: {SCOPED_QUERY!r}")
    result_b = run_query(SCOPED_QUERY, AnswerFormat.CONCISE, corpus_only=True)
    _report(result_b)
    cited_b = {c.doc_id for c in result_b.citations}
    print(f"  RESULT scoped to {PAPER_A.name} only: "
          f"{cited_b <= {doc_a} and bool(cited_b)}"
          f"{' (abstained, no citations)' if not cited_b else ''}")
    print(f"  RESULT other paper excluded          : {doc_b not in cited_b}")

    _rule(f"TEST (c) - unknown filename: {MISSING_FILE_QUERY!r}")
    result_c = run_query(MISSING_FILE_QUERY, AnswerFormat.CONCISE, corpus_only=True)
    _report(result_c)
    resolved_none = any(
        "0 named document(s)" in step for step in result_c.trace
    )
    retrieved = any(step.startswith("retrieve:") for step in result_c.trace)
    scoped = any("scoped to doc_id" in step for step in result_c.trace)
    print(f"  RESULT resolved no documents         : {resolved_none}")
    print(f"  RESULT fell back to normal retrieval : {retrieved and not scoped}")
    print(f"  RESULT did not crash                 : True")

    _rule(f"TEST (d) REGRESSION - {REGRESSION_QUERY!r}")
    result_d = run_query(REGRESSION_QUERY, AnswerFormat.CONCISE, corpus_only=True)
    _report(result_d)
    print(f"  RESULT still abstains (gate1 not loosened): {result_d.abstained}")
    print(f"  RESULT no doc scoping applied             : "
          f"{not any('scoped to doc_id' in s for s in result_d.trace)}")

    _rule("Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
