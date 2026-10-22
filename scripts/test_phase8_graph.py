"""Phase 8 smoke test: the full LangGraph pipeline end to end.

Ingests tests/fixtures/ into ISOLATED temp stores, then runs run_query()
across four cases chosen to exercise every branch of the control flow:
verified answers, corpus-only abstention, and abstention after the one
permitted external fetch.

    python scripts/test_phase8_graph.py
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

# Redirect both stores to throwaway temp locations BEFORE anything reads them.
from verirag.config import settings  # noqa: E402

_TMP_DIR = Path(tempfile.mkdtemp(prefix="verirag_phase8_"))
settings.chroma_persist_dir = str(_TMP_DIR / "chroma")
settings.bm25_corpus_path = str(_TMP_DIR / "bm25_corpus.jsonl")

from verirag.generation.generator import AnswerFormat  # noqa: E402
from verirag.graph.build import VeriRAGResult, run_query  # noqa: E402
from verirag.ingest import ingest_directory  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

CASES = [
    (
        "quarterly revenue chart",
        AnswerFormat.CONCISE,
        False,
        "expect verified answer, no regeneration",
    ),
    (
        "two-gate verification critic",
        AnswerFormat.DETAILED,
        False,
        "expect verified answer",
    ),
    (
        "weather forecast tomorrow",
        AnswerFormat.CONCISE,
        True,
        "expect abstain (coverage), NO fetch attempt - corpus_only blocks it",
    ),
    (
        "weather forecast tomorrow",
        AnswerFormat.CONCISE,
        False,
        "expect ONE fetch attempt (stub), then abstain (coverage)",
    ),
]


def _report(result: VeriRAGResult) -> None:
    print(f"-- TRACE ({len(result.trace)} steps) --")
    for i, step in enumerate(result.trace, start=1):
        print(f"  {i:2}. {step}")

    print(f"-- OUTCOME --")
    print(f"  abstained          : {result.abstained}")
    print(f"  regeneration_count : {result.regeneration_count}")

    if result.abstained:
        print("  abstention_reason  :")
        for line in (result.abstention_reason or "").splitlines():
            print(f"      {line}")
        print(f"  final_answer       : {result.answer}")
    else:
        print("  ANSWER:")
        for line in (result.answer or "").splitlines():
            print(f"      {line}")
        print(f"  CITATIONS ({len(result.citations)}):")
        for citation in result.citations:
            pages = [p for p in citation.pages if p is not None] or "n/a"
            print(
                f"    [{citation.number}] {Path(citation.source_path).name} "
                f"| pages={pages} | chunks={citation.contributing_chunk_ids}"
            )

    # Branch assertions, reported rather than asserted so all cases still run.
    fetch_steps = [s for s in result.trace if s.startswith("fetch_external:")]
    fetch_attempts = len([s for s in fetch_steps if "returned" in s])
    print(f"  external fetch attempts: {fetch_attempts} (bound is 1)")


def main() -> None:
    print(f"Isolated stores under: {_TMP_DIR}")
    print(
        f"Bounds: max_critic_iterations={settings.max_critic_iterations}, "
        f"max_llm_calls_per_query={settings.max_llm_calls_per_query}"
    )
    print("=" * 76)
    print("INGEST tests/fixtures/")
    for result in ingest_directory(FIXTURES_DIR):
        print(f"  {result}")

    for i, (query, answer_format, corpus_only, expectation) in enumerate(
        CASES, start=1
    ):
        print("=" * 76)
        print(f"CASE {i}: query={query!r}")
        print(
            f"  format={answer_format.value}  corpus_only={corpus_only}"
        )
        print(f"  ({expectation})")
        print("=" * 76)
        _report(run_query(query, answer_format, corpus_only=corpus_only))

    print("=" * 76)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
