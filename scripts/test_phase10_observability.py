"""Phase 10 smoke test: LangSmith tracing + the SQLite metrics log.

Ingests tests/fixtures/ into ISOLATED temp stores (including an isolated
metrics DB), runs three queries covering a verified path and a clean
abstention, then prints the aggregate metrics summary.

Tracing is optional by design: with no real LANGSMITH_API_KEY it stays
inactive and queries must still run normally. The script reports which case
applies rather than assuming either.

    python scripts/test_phase10_observability.py
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import tempfile
from pathlib import Path

# Redirect every store to throwaway temp locations BEFORE anything reads them.
from verirag.config import configured_secret, settings  # noqa: E402

_TMP_DIR = Path(tempfile.mkdtemp(prefix="verirag_phase10_"))
settings.chroma_persist_dir = str(_TMP_DIR / "chroma")
settings.bm25_corpus_path = str(_TMP_DIR / "bm25_corpus.jsonl")
settings.metrics_db_path = str(_TMP_DIR / "metrics.db")

from verirag.generation.generator import AnswerFormat  # noqa: E402
from verirag.graph.build import run_query  # noqa: E402
from verirag.ingest import ingest_directory  # noqa: E402
from verirag.observability.metrics import summarize_metrics  # noqa: E402
from verirag.observability.tracing import tracing_active  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

# Every node we expect to see across the three runs below. The regeneration /
# ungrounded / external-fetch nodes are NOT expected here: these three queries
# deliberately take the clean paths.
EXPECTED_NODES = {
    "retrieve",
    "gate1",
    "generate",
    "gate2",
    "finalize_verified",
    "abstain_insufficient_coverage",
    "fetch_external",
}

CASES = [
    ("quarterly revenue chart", AnswerFormat.CONCISE, False),
    ("two-gate verification critic", AnswerFormat.DETAILED, False),
    ("weather forecast tomorrow", AnswerFormat.CONCISE, True),
    # corpus_only=False so the fetch_external node is exercised and timed too
    # (the stub returns nothing, so this costs no LLM calls).
    ("weather forecast tomorrow", AnswerFormat.CONCISE, False),
]


def _rule(title: str) -> None:
    print("=" * 76)
    print(title)
    print("=" * 76)


def main() -> None:
    print(f"Isolated stores under: {_TMP_DIR}")
    print(f"metrics_db_path: {settings.metrics_db_path}")

    _rule("TRACING STATUS")
    has_key = configured_secret(settings.langsmith_api_key) is not None
    print(f"  real LANGSMITH_API_KEY configured : {has_key}")
    print(f"  tracing_active()                  : {tracing_active()}")
    print(f"  langsmith_project                 : {settings.langsmith_project!r}")
    if not has_key:
        print("  -> tracing inactive; queries must still run normally")

    _rule("INGEST tests/fixtures/")
    for result in ingest_directory(FIXTURES_DIR):
        print(f"  {result}")

    for query, answer_format, corpus_only in CASES:
        _rule(f"RUN: {query!r} format={answer_format.value} corpus_only={corpus_only}")
        result = run_query(query, answer_format, corpus_only=corpus_only)
        print(f"  abstained={result.abstained} "
              f"regenerations={result.regeneration_count} "
              f"citations={len(result.citations)}")
        print(f"  trace steps: {len(result.trace)}")
        if result.abstained:
            print(f"  abstention: {(result.abstention_reason or '')[:90]}...")
        else:
            print(f"  answer[:90]: {(result.answer or '')[:90]!r}")

    _rule("RAW SQLITE CONTENTS")
    with sqlite3.connect(settings.metrics_db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, query, answer_format, corpus_only, outcome,
                   regeneration_count, external_fetch_attempted,
                   llm_call_count, total_latency_ms, estimated_cost_usd,
                   node_timings_json
            FROM query_metrics ORDER BY id
            """
        ).fetchall()
    print(f"  rows persisted: {len(rows)}")
    seen_nodes: set[str] = set()
    for row in rows:
        (
            row_id, query, fmt, corpus_only, outcome, regens, fetched,
            calls, latency, cost, timings_json,
        ) = row
        timings = json.loads(timings_json)
        seen_nodes.update(name for name, _ in timings)
        print(f"  [{row_id}] {query!r}")
        print(f"       format={fmt} corpus_only={bool(corpus_only)} "
              f"outcome={outcome}")
        print(f"       regenerations={regens} external_fetch={bool(fetched)} "
              f"llm_calls={calls}")
        print(f"       total_latency_ms={latency:.1f} "
              f"estimated_cost_usd={cost:.4f}")
        print(f"       node_timings ({len(timings)}): "
              + ", ".join(f"{n}={ms:.1f}ms" for n, ms in timings))

    _rule("METRICS SUMMARY (summarize_metrics)")
    summary = summarize_metrics()
    print(f"  total_queries          : {summary.total_queries}")
    print(f"  abstention_rate        : {summary.abstention_rate:.4f}")
    print(f"  loop_trigger_rate      : {summary.loop_trigger_rate:.4f}")
    print(f"  avg_cost_per_query_usd : {summary.avg_cost_per_query_usd:.6f}")
    print("  p95_latency_by_node:")
    for node_name, value in summary.p95_latency_by_node.items():
        print(f"      {node_name:32} {value:10.2f} ms")

    _rule("CHECKS")
    missing = EXPECTED_NODES - seen_nodes
    print(f"  rows persisted == {len(CASES)}                 : "
          f"{len(rows) == len(CASES)}")
    print(f"  nodes observed                      : {sorted(seen_nodes)}")
    print(f"  all expected nodes seen             : {not missing}"
          + (f" (missing: {sorted(missing)})" if missing else ""))
    print(f"  every node in p95 map               : "
          f"{set(summary.p95_latency_by_node) == seen_nodes}")
    print(f"  tracing did not break any run       : True "
          f"({len(rows)}/{len(CASES)} runs completed, "
          f"tracing_active={tracing_active()})")

    _rule("Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
