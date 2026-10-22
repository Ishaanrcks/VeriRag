"""Phase 5 smoke test: cited generation over small-to-big context.

Ingests tests/fixtures/ into ISOLATED temp stores, retrieves with the Phase 4
hybrid pipeline, then generates an answer for each query in BOTH formats and
checks that the [n] markers the model actually used line up with the citations
we handed it.

    python scripts/test_phase5_generation.py
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from pathlib import Path

# Redirect both stores to throwaway temp locations BEFORE anything reads them.
from verirag.config import settings  # noqa: E402

_TMP_DIR = Path(tempfile.mkdtemp(prefix="verirag_phase5_"))
settings.chroma_persist_dir = str(_TMP_DIR / "chroma")
settings.bm25_corpus_path = str(_TMP_DIR / "bm25_corpus.jsonl")

from verirag.generation.generator import (  # noqa: E402
    AnswerFormat,
    GeneratedAnswer,
    generate_answer,
)
from verirag.ingest import ingest_directory  # noqa: E402
from verirag.retrieval.hybrid import hybrid_retrieve  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

QUERIES = [
    "quarterly revenue chart",
    "two-gate verification critic",
]

_CITATION_RE = re.compile(r"\[(\d+)\]")


def _check_citations(result: GeneratedAnswer) -> None:
    """Cross-check the [n] markers used in the answer against our citations."""
    used = sorted({int(n) for n in _CITATION_RE.findall(result.answer_text)})
    available = {c.number for c in result.citations}
    dangling = [n for n in used if n not in available]
    unused = sorted(available - set(used))

    print(f"  CONSISTENCY CHECK:")
    print(f"    citation numbers used in answer : {used or '(none)'}")
    print(f"    citation numbers available      : {sorted(available) or '(none)'}")

    flags: list[str] = []
    if not used:
        flags.append("FLAG: answer cited nothing at all")
    if dangling:
        flags.append(f"FLAG: cited non-existent citation number(s) {dangling}")
    if unused:
        # Informational only - not every provided passage must be cited.
        print(f"    (provided but uncited: {unused})")

    if flags:
        for flag in flags:
            print(f"    {flag}")
        print("    result: FLAGGED")
    else:
        print("    result: PASSED (every [n] used maps to a real citation)")


def _run(query: str, answer_format: AnswerFormat) -> None:
    print("=" * 76)
    print(f"QUERY: {query!r}   FORMAT: {answer_format.value}")
    print("=" * 76)

    hits = hybrid_retrieve(query)
    result = generate_answer(query, hits, answer_format)

    print("-- ANSWER --")
    print(result.answer_text)

    print(f"-- CITATIONS ({len(result.citations)}) --")
    for citation in result.citations:
        pages = [p for p in citation.pages if p is not None] or "n/a"
        print(
            f"  [{citation.number}] {citation.source_path} "
            f"| pages={pages} | doc_id={citation.doc_id}"
        )
        print(f"       contributing chunk_ids: {citation.contributing_chunk_ids}")

    _check_citations(result)


def main() -> None:
    print(f"Isolated stores under: {_TMP_DIR}")
    print("=" * 76)
    print("INGEST tests/fixtures/")
    for result in ingest_directory(FIXTURES_DIR):
        print(f"  {result}")

    for query in QUERIES:
        for answer_format in (AnswerFormat.CONCISE, AnswerFormat.DETAILED):
            _run(query, answer_format)

    print("=" * 76)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
