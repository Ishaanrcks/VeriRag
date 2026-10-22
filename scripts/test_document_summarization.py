"""Test the whole-document summarization path.

Reproduces the live failure that motivated this fix: naming an ingested file
and asking for a summary used to embed the filename plus the instruction as a
search query, match nothing (observed rerank score 0.0036), and abstain for
insufficient coverage. It should now read the whole document in order and
return a grounded, cited summary.

Uses an ISOLATED temp store but ingests a REAL uploaded PDF, so the document
is long enough to exercise the map-reduce path rather than a single call.

    python scripts/test_document_summarization.py
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

# Redirect every store to throwaway temp locations BEFORE anything reads them.
from verirag.config import settings  # noqa: E402

_TMP_DIR = Path(tempfile.mkdtemp(prefix="verirag_summary_"))
settings.chroma_persist_dir = str(_TMP_DIR / "chroma")
settings.bm25_corpus_path = str(_TMP_DIR / "bm25_corpus.jsonl")
settings.metrics_db_path = str(_TMP_DIR / "metrics.db")

from verirag.generation.generator import AnswerFormat  # noqa: E402
from verirag.graph.build import run_query  # noqa: E402
from verirag.ingest import ingest_file  # noqa: E402
from verirag.retrieval.vector_store import get_chunks_by_doc_id  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "data" / "raw" / "uploads" / "2311.14966v1.pdf"
QUERY = "summarize this paper please: 2311.14966v1.pdf"


def _rule(title: str) -> None:
    print("=" * 76)
    print(title)
    print("=" * 76)


def main() -> None:
    print(f"Isolated stores under: {_TMP_DIR}")

    if not PAPER.exists():
        print(f"SKIP: {PAPER} not found. Upload it first (data/raw/uploads/).")
        return

    _rule(f"INGEST {PAPER.name} (real PDF; this takes a few minutes)")
    result = ingest_file(PAPER)
    print(f"  {result}")
    doc_id = result.doc_id

    chunks = get_chunks_by_doc_id(doc_id)
    indices = [c.metadata.get("chunk_index") for c in chunks]
    total_chars = sum(len(c.raw_text) for c in chunks)
    print(f"  chunks retrievable in order: {len(chunks)}")
    print(f"  chunk_index sequence correct: {indices == sorted(indices)}")
    print(f"  first 10 indices: {indices[:10]}")
    print(f"  total ordered text: {total_chars} chars "
          f"(single-pass budget {settings.summarization_context_max_chars})")
    print(f"  expects map-reduce: {total_chars > settings.summarization_context_max_chars}")

    _rule(f"TEST (a) - run_query({QUERY!r})")
    run_result = run_query(QUERY, AnswerFormat.DETAILED, corpus_only=True)

    print(f"-- TRACE ({len(run_result.trace)} steps) --")
    for i, step in enumerate(run_result.trace, start=1):
        print(f"  {i:2}. {step}")

    print("-- OUTCOME --")
    print(f"  abstained          : {run_result.abstained}")
    print(f"  regeneration_count : {run_result.regeneration_count}")
    if run_result.abstained:
        print("  abstention_reason  :")
        for line in (run_result.abstention_reason or "").splitlines():
            print(f"      {line}")
    else:
        print("  SUMMARY:")
        for line in (run_result.answer or "").splitlines():
            print(f"      {line}")

    print(f"  CITATIONS ({len(run_result.citations)}):")
    cited_doc_ids = set()
    for citation in run_result.citations:
        cited_doc_ids.add(citation.doc_id)
        print(f"    [{citation.number}] doc_id={citation.doc_id} "
              f"chunks={len(citation.contributing_chunk_ids)} "
              f"source={Path(str(citation.source_path)).name}")

    _rule("CHECKS")
    took_summary_path = any(
        step.startswith("summarize_document:") for step in run_result.trace
    )
    skipped_gate1 = not any(step.startswith("gate1:") for step in run_result.trace)
    ran_gate2 = any(step.startswith("gate2:") for step in run_result.trace)
    print(f"  routed to summarize_document path   : {took_summary_path}")
    print(f"  skipped gate1 (coverage self-evident): {skipped_gate1}")
    print(f"  still ran gate2 (invariant preserved): {ran_gate2}")
    print(f"  produced a grounded summary          : {not run_result.abstained}")
    print(f"  cites only the named document        : "
          f"{cited_doc_ids == {doc_id} if cited_doc_ids else 'n/a (abstained)'}")

    _rule("Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
