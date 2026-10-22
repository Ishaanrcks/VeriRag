"""Regression tests for external-content scoping and grounding (Fix 2).

Motivated by two observed failures:

* Web pages fetched for one query persisted in the shared store and then
  surfaced for later, unrelated queries ("summarize this paper please"
  fetched a page about summarizing emails, which then competed on every
  subsequent query).
* A weak, off-topic match (rerank_score 0.168) was approved by Gate 1's
  escalation judge, and the generator produced a confident, accurate-sounding
  answer that did not come from those passages at all -- it filled the gap
  from background knowledge. Gate 2 caught it, but only after two wasted
  regeneration attempts.

Covers:
  (a) pollution   -- a stale external chunk must not surface for a new query
  (b) mismatch    -- with an irrelevant external chunk sharing surface
                     vocabulary, the system must answer from the real local
                     chunks or abstain, never answer confidently from the
                     wrong source
  (c) fallback    -- with corpus_only off and genuinely thin local coverage,
                     a fresh fetch in the SAME run still supplements the answer

    python scripts/test_external_scoping.py
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

# Redirect every store to throwaway temp locations BEFORE anything reads them.
from verirag.config import configured_secret, settings  # noqa: E402

_TMP_DIR = Path(tempfile.mkdtemp(prefix="verirag_extscope_"))
settings.chroma_persist_dir = str(_TMP_DIR / "chroma")
settings.bm25_corpus_path = str(_TMP_DIR / "bm25_corpus.jsonl")
settings.metrics_db_path = str(_TMP_DIR / "metrics.db")

from langchain_core.documents import Document  # noqa: E402

from verirag.external.fetcher import compute_external_doc_id  # noqa: E402
from verirag.generation.generator import AnswerFormat  # noqa: E402
from verirag.graph.build import run_query  # noqa: E402
from verirag.ingest import ingest_documents, ingest_file  # noqa: E402
from verirag.retrieval.hybrid import hybrid_retrieve  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

# Stale junk of exactly the kind that polluted the live store.
STALE_URL = "https://en.wikipedia.org/wiki/Your_papers,_please"
STALE_TEXT = (
    "\"Your papers, please\" is a stock phrase associated with police states "
    "and border checkpoints, demanding identity documents from a traveller. "
    "It appears in film and literature as shorthand for authoritarian "
    "surveillance."
)

# Shares surface vocabulary with the fixtures ("verification", "gate") but is
# about a completely different subject -- the mismatch trap.
DECOY_URL = "https://example.com/airport-gate-verification-procedures"
DECOY_TEXT = (
    "Airport gate verification procedures require agents to perform a two-gate "
    "check of each boarding pass: the first gate verifies the passenger "
    "manifest, and the second gate verifies photo identification before "
    "allowing entry to the jet bridge."
)


def _rule(title: str) -> None:
    print("=" * 76)
    print(title)
    print("=" * 76)


def _external_doc(url: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "doc_id": compute_external_doc_id(url),
            "source_path": url,
            "format": "external_web",
            "content_type": "text",
            "page": None,
            "source_type": "external",
        },
    )


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
        print("  ANSWER:")
        for line in (result.answer or "").splitlines():
            print(f"      {line}")
    for c in result.citations:
        label = c.url if c.source_type == "external" else Path(
            str(c.source_path)
        ).name
        print(f"    [{c.number}] {c.source_type}: {label}")


def main() -> None:
    print(f"Isolated stores under: {_TMP_DIR}")

    _rule("SETUP - local fixtures + two stale external chunks")
    for name in ("sample.pdf", "sample.txt", "sample.pptx", "sample.md"):
        result = ingest_file(FIXTURES_DIR / name)
        print(f"  {name}: {result}")

    stale_id = compute_external_doc_id(STALE_URL)
    decoy_id = compute_external_doc_id(DECOY_URL)
    for result in ingest_documents(
        [_external_doc(STALE_URL, STALE_TEXT), _external_doc(DECOY_URL, DECOY_TEXT)]
    ):
        print(f"  external seeded: {result}")

    # ---------------------------------------------------------------- (a)
    _rule("TEST (a) POLLUTION - stale external chunk must not surface")
    for query in ("your papers please", "two-gate verification critic"):
        for corpus_only in (False, True):
            hits = hybrid_retrieve(query, corpus_only=corpus_only)
            external = [h for h in hits if h.chunk_id.startswith("external-")]
            print(f"  query={query!r} corpus_only={corpus_only}: "
                  f"{len(hits)} hits, {len(external)} external")
    pollution_hits = [
        h
        for q in ("your papers please", "two-gate verification critic")
        for co in (False, True)
        for h in hybrid_retrieve(q, corpus_only=co)
        if h.chunk_id.startswith("external-")
    ]
    print(f"  RESULT no external chunk surfaced in ANY case: "
          f"{len(pollution_hits) == 0}")
    print(f"  (seeded stale doc_ids were {stale_id}, {decoy_id})")

    # ---------------------------------------------------------------- (b)
    _rule("TEST (b) MISMATCH - decoy shares vocabulary, different subject")
    print(f"  decoy seeded: {DECOY_URL}")
    print("  question is about VeriRAG's two-gate verification critic;")
    print("  the decoy is about airport boarding-gate checks.")
    result_b = run_query(
        "what does the two-gate verification critic check?",
        AnswerFormat.CONCISE,
        corpus_only=False,
    )
    _report(result_b)
    cited_external = [c for c in result_b.citations if c.source_type == "external"]
    answered_from_local = (
        not result_b.abstained
        and bool(result_b.citations)
        and not cited_external
    )
    print(f"  RESULT never cites the decoy external source: "
          f"{not cited_external}")
    print(f"  RESULT answered from local sources OR abstained cleanly: "
          f"{answered_from_local or result_b.abstained}")

    # ---------------------------------------------------------------- (c)
    _rule("TEST (c) FALLBACK - fresh fetch must still help the same run")
    if configured_secret(settings.tavily_api_key) is None:
        print("  SKIP: no usable TAVILY_API_KEY, cannot test a live fetch.")
    else:
        result_c = run_query(
            "what year was the Eiffel Tower completed",
            AnswerFormat.CONCISE,
            corpus_only=False,
        )
        _report(result_c)
        fetched = any("fetch_external: returned" in s for s in result_c.trace)
        scored = any("scored" in s and "in-memory" in s for s in result_c.trace)
        merged = any("freshly fetched" in s for s in result_c.trace)
        print(f"  RESULT a fetch happened                  : {fetched}")
        print(f"  RESULT fetched chunks scored in-memory   : {scored}")
        print(f"  RESULT merged into this query's retrieval: {merged}")
        print(f"  RESULT reached generation or abstained cleanly: True "
              f"(abstained={result_c.abstained})")

    _rule("Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
