"""Phase 6 smoke test: Gate 1 coverage check.

Ingests tests/fixtures/ into ISOLATED temp stores, retrieves with the Phase 4
hybrid pipeline, then runs check_coverage() over a range of queries.

A counting wrapper is installed around Gate 1's chat_completion so we can
*prove* the threshold paths are cheap (zero LLM calls) rather than assume it.
The LLM-escalation path is then exercised directly, in both parse directions.

    python scripts/test_phase6_gate1.py
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path

# Redirect both stores to throwaway temp locations BEFORE anything reads them.
from verirag.config import settings  # noqa: E402

_TMP_DIR = Path(tempfile.mkdtemp(prefix="verirag_phase6_"))
settings.chroma_persist_dir = str(_TMP_DIR / "chroma")
settings.bm25_corpus_path = str(_TMP_DIR / "bm25_corpus.jsonl")

import verirag.critic.gate1_coverage as gate1  # noqa: E402
from verirag.critic.gate1_coverage import check_coverage  # noqa: E402
from verirag.ingest import ingest_directory  # noqa: E402
from verirag.retrieval.hybrid import hybrid_retrieve  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

# --- instrument Gate 1's LLM usage so "no LLM call" is verified, not assumed ---
_real_chat_completion = gate1.chat_completion
_llm_calls = {"count": 0}


def _counting_chat_completion(*args, **kwargs):
    _llm_calls["count"] += 1
    return _real_chat_completion(*args, **kwargs)


gate1.chat_completion = _counting_chat_completion

CASES = [
    ("quarterly revenue chart", "expect sufficient via threshold_high"),
    ("two-gate verification critic", "expect sufficient via threshold_high"),
    ("weather forecast tomorrow", "expect insufficient via threshold_low"),
    ("artificial intelligence systems", "borderline candidate - report actual"),
]


def _report(verdict: gate1.Gate1Verdict, llm_calls: int, elapsed_ms: float) -> None:
    print(f"  max_rerank_score : {verdict.max_rerank_score:.6f}")
    print(f"  score_spread     : {verdict.score_spread:.6f}")
    print(f"  decision_path    : {verdict.decision_path}")
    print(f"  verdict          : {verdict.verdict}")
    print(f"  llm_reason       : {verdict.llm_reason or '(none)'}")
    print(f"  LLM calls made   : {llm_calls}")
    print(f"  elapsed          : {elapsed_ms:.1f} ms")


def _run_case(query: str, expectation: str) -> None:
    print("=" * 76)
    print(f"QUERY: {query!r}")
    print(f"  ({expectation})")
    hits = hybrid_retrieve(query)
    print(f"  hits retrieved   : {len(hits)}")

    before = _llm_calls["count"]
    start = time.perf_counter()
    verdict = check_coverage(query, hits)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _report(verdict, _llm_calls["count"] - before, elapsed_ms)


def _run_direct_escalation() -> None:
    """Exercise _llm_judge_coverage directly, in both parse directions."""
    print("=" * 76)
    print("DIRECT ESCALATION TEST - _llm_judge_coverage() with real hits")
    print("=" * 76)

    hits = hybrid_retrieve("two-gate verification critic")[:3]
    print(f"  using {len(hits)} real hits:")
    for hit in hits:
        print(
            f"    - {hit.chunk_id} (rerank={hit.rerank_score:.4f}): "
            f"{hit.raw_text[:70]!r}"
        )

    probes = [
        (
            "What does Gate 1 check?",
            "covered by these passages -> expect SUFFICIENT",
        ),
        (
            "What exact numeric score threshold does Gate 1 use to decide?",
            "NOT in the corpus -> expect INSUFFICIENT",
        ),
    ]

    for query, expectation in probes:
        print(f"\n  synthetic query: {query!r}")
        print(f"    ({expectation})")
        start = time.perf_counter()
        verdict, reason = gate1._llm_judge_coverage(query, hits)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        parsed_ok = verdict in ("sufficient", "insufficient")
        print(f"    parsed verdict : {verdict}")
        print(f"    reason         : {reason}")
        print(f"    parsed cleanly : {parsed_ok}")
        print(f"    elapsed        : {elapsed_ms:.1f} ms")


def main() -> None:
    print(f"Isolated stores under: {_TMP_DIR}")
    print(
        f"Gate 1 thresholds: low={settings.gate1_low_threshold} "
        f"high={settings.gate1_high_threshold}"
    )
    print("=" * 76)
    print("INGEST tests/fixtures/")
    for result in ingest_directory(FIXTURES_DIR):
        print(f"  {result}")

    for query, expectation in CASES:
        _run_case(query, expectation)

    _run_direct_escalation()

    print("=" * 76)
    print(f"Total Gate 1 LLM calls across all threshold cases + escalations: "
          f"{_llm_calls['count']}")
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
