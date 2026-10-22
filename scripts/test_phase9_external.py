"""Phase 9 smoke test: real external fetch, ranking penalty, corroboration.

Parts 1-3 exercise the live path (direct fetch, then two full graph runs).
Without a configured TAVILY_API_KEY these legitimately degrade to "no external
material -> abstain", which is itself the behaviour worth confirming.

Part 4 then validates the two pieces of NEW POLICY logic offline, with clearly
synthetic external material, so the corroboration rule and the ranking penalty
are actually exercised regardless of whether web access is available.

    python scripts/test_phase9_external.py
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

# Redirect both stores to throwaway temp locations BEFORE anything reads them.
from verirag.config import settings  # noqa: E402

_TMP_DIR = Path(tempfile.mkdtemp(prefix="verirag_phase9_"))
settings.chroma_persist_dir = str(_TMP_DIR / "chroma")
settings.bm25_corpus_path = str(_TMP_DIR / "bm25_corpus.jsonl")

from verirag.critic.gate2_groundedness import check_groundedness  # noqa: E402
from verirag.external.fetcher import (  # noqa: E402
    _configured_api_key,
    compute_external_doc_id,
    fetch_external,
)
from verirag.generation.generator import (  # noqa: E402
    AnswerFormat,
    GeneratedAnswer,
    build_context_block,
)
from verirag.graph.build import VeriRAGResult, run_query  # noqa: E402
from verirag.ingest import ingest_directory  # noqa: E402
from verirag.retrieval.hybrid import _apply_external_penalty  # noqa: E402
from verirag.retrieval.types import SearchHit  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

STABLE_QUERY = "what year was the Eiffel Tower completed"
HARD_QUERY = "weather forecast tomorrow"

URL_A = "https://example.com/eiffel-tower-history"
URL_B = "https://example.org/paris-landmarks"
CLAIM_TEXT = "The Eiffel Tower was completed in 1889."
TEXT_A = (
    "The Eiffel Tower was completed in 1889 for the World's Fair held in "
    "Paris that year."
)
TEXT_B = (
    "Construction of the Eiffel Tower finished in 1889, in time for the "
    "Exposition Universelle."
)


def _rule(title: str) -> None:
    print("=" * 76)
    print(title)
    print("=" * 76)


def _external_hit(chunk_id: str, url: str, text: str, score: float) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        raw_text=text,
        metadata={
            "doc_id": compute_external_doc_id(url),
            "source_path": url,
            "format": "external_web",
            "content_type": "text",
            "page": None,
            "source_type": "external",
            "parent_text": text,
        },
        rerank_score=score,
    )


def _local_hit(chunk_id: str, text: str, score: float) -> SearchHit:
    # Note: no source_type key at all -- that is how local Documents look.
    return SearchHit(
        chunk_id=chunk_id,
        raw_text=text,
        metadata={
            "doc_id": "local-doc",
            "source_path": "/corpus/local.txt",
            "format": "txt",
            "content_type": "text",
            "page": None,
            "parent_text": text,
        },
        rerank_score=score,
    )


def _report_result(result: VeriRAGResult) -> None:
    print(f"-- TRACE ({len(result.trace)} steps) --")
    for i, step in enumerate(result.trace, start=1):
        print(f"  {i:2}. {step}")
    print("-- OUTCOME --")
    print(f"  abstained          : {result.abstained}")
    print(f"  regeneration_count : {result.regeneration_count}")
    if result.abstained:
        print("  abstention_reason  :")
        for line in (result.abstention_reason or "").splitlines():
            print(f"      {line}")
    else:
        print("  ANSWER:")
        for line in (result.answer or "").splitlines():
            print(f"      {line}")
    print(f"  CITATIONS ({len(result.citations)}):")
    for c in result.citations:
        label = Path(c.source_path).name if c.source_type == "local" else c.url
        print(f"    [{c.number}] source_type={c.source_type} {label}")


def _report_gate2(answer: GeneratedAnswer, title: str) -> None:
    print(f"  {title}")
    verdict = check_groundedness(answer)
    for claim in verdict.claims:
        prob = (
            "None"
            if claim.entailment_prob is None
            else f"{claim.entailment_prob:.6f}"
        )
        print(f"    claim_type={claim.claim_type} verdict={claim.verdict}")
        print(f"      text  : {claim.claim_text}")
        print(f"      prob  : {prob}")
        print(f"      chunks: {claim.evidence_chunk_ids or '(none)'}")
    print(f"    overall_grounded = {verdict.overall_grounded}")
    print(f"    blocking claims  = {len(verdict.unsupported_claims)}")


def part1_direct_fetch() -> None:
    _rule(f"PART 1 - direct fetch_external({STABLE_QUERY!r})")
    # Use the fetcher's own check so a leftover `tvly-...` placeholder in .env
    # is reported as unset rather than as configured.
    key_state = "configured" if _configured_api_key() else "unset/placeholder"
    print(f"  tavily_api_key: {key_state}")
    documents = fetch_external(STABLE_QUERY)
    print(f"  returned {len(documents)} document(s)")
    for doc in documents:
        print(f"    doc_id      : {doc.metadata['doc_id']}")
        print(f"    source_path : {doc.metadata['source_path']}")
        print(f"    format      : {doc.metadata['format']}")
        print(f"    source_type : {doc.metadata['source_type']}")
        print(f"    content[:200]: {doc.page_content[:200]!r}")
    if not documents:
        print("  -> empty list returned cleanly (no exception raised)")


def part2_and_3_graph_runs() -> None:
    for query in (STABLE_QUERY, HARD_QUERY):
        _rule(f"FULL GRAPH RUN - {query!r} (corpus_only=False)")
        _report_result(run_query(query, AnswerFormat.CONCISE, corpus_only=False))


def part4_policy_validation() -> None:
    _rule("PART 4 - OFFLINE POLICY VALIDATION (synthetic external material)")

    print("-- 4a. ranking penalty (_apply_external_penalty) --")
    print(f"  penalty factor = {settings.external_source_score_penalty}")
    mixed = [
        _external_hit("ext-1", URL_A, TEXT_A, 0.95),
        _local_hit("loc-1", "Local document about the tower.", 0.90),
    ]
    print("  before: " + ", ".join(
        f"{h.chunk_id}={h.rerank_score:.4f}" for h in mixed
    ))
    ranked = _apply_external_penalty(mixed)
    print("  after : " + ", ".join(
        f"{h.chunk_id}={h.rerank_score:.4f}" for h in ranked
    ))
    print(f"  top hit is now: {ranked[0].chunk_id}")
    print(
        "  external discounted 0.95 -> "
        f"{0.95 * settings.external_source_score_penalty:.4f}, "
        f"local (0.90) now ranks first: {ranked[0].chunk_id == 'loc-1'}"
    )

    only_external = [_external_hit("ext-9", URL_A, TEXT_A, 0.40)]
    survived = _apply_external_penalty(only_external)
    print(
        "  external-only still surfaces (discounted, never excluded): "
        f"{len(survived) == 1}, score={survived[0].rerank_score:.4f}"
    )

    print("-- 4b. corroboration rule --")

    # (i) ONE external source -> expect external_unsupported
    same_page = [
        _external_hit("extA-0", URL_A, TEXT_A, 0.9),
        _external_hit("extA-1", URL_A, TEXT_A, 0.8),
    ]
    _, citations = build_context_block(same_page)
    _report_gate2(
        GeneratedAnswer(
            answer_text=f"{CLAIM_TEXT[:-1]} [1].",
            citations=citations,
            format_used=AnswerFormat.CONCISE,
            hits_used=same_page,
        ),
        "(i) 1 distinct external URL (2 chunks, same page) "
        "-> expect external_unsupported",
    )

    # (ii) TWO distinct external sources -> expect supported_combination
    two_sources = [
        _external_hit("extA-0", URL_A, TEXT_A, 0.9),
        _external_hit("extB-0", URL_B, TEXT_B, 0.8),
    ]
    _, citations = build_context_block(two_sources)
    _report_gate2(
        GeneratedAnswer(
            answer_text=f"{CLAIM_TEXT[:-1]} [1][2].",
            citations=citations,
            format_used=AnswerFormat.CONCISE,
            hits_used=two_sources,
        ),
        "(ii) 2 distinct external URLs -> expect supported_combination",
    )

    # (iii) local + external -> local alone grounds it
    mixed_evidence = [
        _local_hit("loc-0", TEXT_A, 0.9),
        _external_hit("extB-0", URL_B, TEXT_B, 0.8),
    ]
    _, citations = build_context_block(mixed_evidence)
    _report_gate2(
        GeneratedAnswer(
            answer_text=f"{CLAIM_TEXT[:-1]} [1][2].",
            citations=citations,
            format_used=AnswerFormat.CONCISE,
            hits_used=mixed_evidence,
        ),
        "(iii) local + external evidence -> expect supported_single (local wins)",
    )

    print("-- 4c. external context block formatting --")
    block, _ = build_context_block(
        [_external_hit("extA-0", URL_A, TEXT_A, 0.9)]
    )
    for line in block.splitlines():
        print(f"    {line}")


def main() -> None:
    print(f"Isolated stores under: {_TMP_DIR}")
    print(
        f"external_max_results={settings.external_max_results} "
        f"search_depth={settings.tavily_search_depth!r} "
        f"penalty={settings.external_source_score_penalty}"
    )
    print("=" * 76)
    print("INGEST tests/fixtures/ (local corpus has nothing on these topics)")
    for result in ingest_directory(FIXTURES_DIR):
        print(f"  {result}")

    part1_direct_fetch()
    part2_and_3_graph_runs()
    part4_policy_validation()

    _rule("Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
