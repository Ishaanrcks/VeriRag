"""Phase 7 smoke test: Gate 2 groundedness check.

Ingests tests/fixtures/ into ISOLATED temp stores, generates DETAILED answers
for the two Phase 5 queries that produced real absence/meta claims, and runs
check_groundedness() on each.

Then the critical test: a synthetic answer built on REAL hits and citations but
containing a deliberately FABRICATED claim ("Revenue was highest in Q1 at 52
million dollars" -- wrong quarter and wrong number per the fixture chart),
alongside one true claim. Gate 2 must flag the false one as not_supported while
still supporting the true one.

    python scripts/test_phase7_gate2.py
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

# Redirect both stores to throwaway temp locations BEFORE anything reads them.
from verirag.config import settings  # noqa: E402

_TMP_DIR = Path(tempfile.mkdtemp(prefix="verirag_phase7_"))
settings.chroma_persist_dir = str(_TMP_DIR / "chroma")
settings.bm25_corpus_path = str(_TMP_DIR / "bm25_corpus.jsonl")

from verirag.critic.gate2_groundedness import (  # noqa: E402
    Gate2Verdict,
    check_groundedness,
)
from verirag.generation.generator import (  # noqa: E402
    AnswerFormat,
    GeneratedAnswer,
    build_context_block,
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

FABRICATED_ANSWER_TEXT = (
    "Revenue was highest in Q1 at 52 million dollars [1]. "
    'The bar chart is titled "Quarterly Revenue 2025" [1].'
)


def _prob(value: float | None) -> str:
    return "None" if value is None else f"{value:.6f}"


def _report(verdict: Gate2Verdict) -> None:
    print(f"-- CLAIMS ({len(verdict.claims)}) --")
    for i, claim in enumerate(verdict.claims, start=1):
        print(f"  [{i}] claim_type={claim.claim_type}  verdict={claim.verdict}")
        print(f"      text            : {claim.claim_text}")
        print(f"      entailment_prob : {_prob(claim.entailment_prob)}")
        print(f"      evidence_chunks : {claim.evidence_chunk_ids or '(none)'}")

    print(f"-- OVERALL GROUNDED: {verdict.overall_grounded} --")
    print(f"-- UNSUPPORTED CLAIMS ({len(verdict.unsupported_claims)}) --")
    if not verdict.unsupported_claims:
        print("  (none)")
    for claim in verdict.unsupported_claims:
        print(f"  - {claim.claim_text}")
        print(
            f"    best entailment_prob={_prob(claim.entailment_prob)} "
            f"checked against {claim.evidence_chunk_ids}"
        )


def _run_real(query: str) -> None:
    print("=" * 76)
    print(f"REAL ANSWER - QUERY: {query!r} (DETAILED)")
    print("=" * 76)
    hits = hybrid_retrieve(query)
    answer = generate_answer(query, hits, AnswerFormat.DETAILED)
    print("-- ANSWER TEXT --")
    print(answer.answer_text)
    _report(check_groundedness(answer))


def _run_fabricated() -> None:
    print("=" * 76)
    print("FABRICATED ANSWER - injected false claim (the critical test)")
    print("=" * 76)

    # Real retrieval and real citations; only the answer text is fabricated.
    hits = hybrid_retrieve("quarterly revenue chart")
    _, citations = build_context_block(hits)
    answer = GeneratedAnswer(
        answer_text=FABRICATED_ANSWER_TEXT,
        citations=citations,
        format_used=AnswerFormat.CONCISE,
        hits_used=hits,
    )

    print("-- ANSWER TEXT (fabricated) --")
    print(answer.answer_text)
    print("   ground truth per fixture chart: Q1=23, Q2=45, Q3=31, Q4=52")
    print("   -> 'highest in Q1 at 52 million' is wrong on BOTH quarter and value")
    _report(check_groundedness(answer))


def main() -> None:
    print(f"Isolated stores under: {_TMP_DIR}")
    print(
        f"Gate 2 threshold: entailment >= {settings.gate2_entailment_threshold} "
        f"| combination cap {settings.gate2_combination_max_chars} chars"
    )
    print("=" * 76)
    print("INGEST tests/fixtures/")
    for result in ingest_directory(FIXTURES_DIR):
        print(f"  {result}")

    for query in QUERIES:
        _run_real(query)

    _run_fabricated()

    print("=" * 76)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
