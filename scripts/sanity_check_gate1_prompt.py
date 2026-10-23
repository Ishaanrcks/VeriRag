"""Cheap sanity check for the Gate 1 judge prompt re-tuning (Bug 2).

Replays only the queries that matter for the prompt edit, against the ALREADY
INGESTED QueensLand eval store, so a bad prompt edit is caught in a couple of
minutes instead of after a full ingestion + eval run.

What it checks:
  Q4  (QuadGraph)  - previously INSUFFICIENT even though the judge's own
                     refusal described the 4 layers. Should now be answerable.
  Q20 (UQ in all)  - false premise, must still be refused.
  Q21 (OPSD->GFM)  - unrelated papers, must still be refused.
  WRONG-TOPIC      - a cross-paper premise built from vocabulary that appears
                     in two different papers; must still be refused.

Read-only with respect to the store (no ingestion).

    python scripts/sanity_check_gate1_prompt.py
"""

from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_STORE = PROJECT_ROOT / "eval_store" / "queensland"

from verirag.config import settings  # noqa: E402

settings.chroma_persist_dir = str(EVAL_STORE / "chroma")
settings.bm25_corpus_path = str(EVAL_STORE / "bm25_corpus.jsonl")
settings.metrics_db_path = str(EVAL_STORE / "sanity_metrics.db")

from verirag.generation.generator import AnswerFormat  # noqa: E402
from verirag.graph.build import run_query  # noqa: E402
from verirag.retrieval.vector_store import get_collection  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

CASES = [
    dict(
        tag="Q4 QuadGraph",
        q="What is QuadGraph and how many layers does it have?",
        want="answerable",
        why="Was a Gate 1 false negative: the refusal reason itself described "
            "the 4 layers.",
    ),
    dict(
        tag="Q20 UQ in all papers",
        q="What uncertainty quantification method do all of these papers use "
          "to produce prediction intervals?",
        want="refused",
        why="False premise - only QpiGNN does UQ.",
    ),
    dict(
        tag="Q21 OPSD -> GFM-RAG",
        q="How does OPSD's on-policy self-distillation improve GFM-RAG's "
          "graph retrieval performance?",
        want="refused",
        why="False premise - unrelated papers, no interaction exists.",
    ),
    dict(
        tag="WRONG-TOPIC dual-head",
        q="How does the dual-head architecture improve GFM-RAG's retrieval "
          "accuracy?",
        want="refused",
        why="Vocabulary bridge: dual-head is QpiGNN's, retrieval is "
            "GFM-RAG's; no such combination exists.",
    ),
]


def _refused(result) -> bool:
    """True if the system declined, by abstention OR an explicit refusal."""
    if result.abstained:
        return True
    text = (result.answer or "").lower()
    return (
        "do not address" in text
        or "do not contain" in text
        or "does not address" in text
        or "not addressed" in text
    )


def main() -> None:
    print(f"eval store: {EVAL_STORE}")
    print(f"chunks available: {get_collection().count()}")
    print()

    passes = 0
    for case in CASES:
        print("=" * 76)
        print(f"{case['tag']}  (want: {case['want']})")
        print(f"  {case['q']}")
        print(f"  rationale: {case['why']}")
        result = run_query(case["q"], AnswerFormat.CONCISE, corpus_only=True)

        refused = _refused(result)
        gate1_steps = [s for s in result.trace if s.startswith("gate1:")]
        for step in gate1_steps:
            print(f"  {step}")
        print(f"  abstained={result.abstained} refused={refused} "
              f"regenerations={result.regeneration_count}")
        if result.abstained:
            print(f"  reason: {(result.abstention_reason or '')[:220]}")
        else:
            print(f"  answer: {(result.answer or '')[:220]}")

        ok = (not refused) if case["want"] == "answerable" else refused
        passes += ok
        print(f"  RESULT: {'PASS' if ok else 'FAIL'}")

    print("=" * 76)
    print(f"{passes}/{len(CASES)} sanity checks passed")
    if passes < len(CASES):
        print("Do NOT start the full eval re-run until these pass.")


if __name__ == "__main__":
    main()
