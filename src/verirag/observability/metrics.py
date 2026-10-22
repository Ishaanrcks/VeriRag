"""SQLite metrics log for query runs.

One row per answered (or abstained) query, so the behaviour the critic is
supposed to produce -- abstention rate, how often the corrective loop fires,
per-node latency, rough cost -- is measurable rather than anecdotal.

``node_timings`` is stored JSON-encoded rather than normalized into a second
table: a node can be visited several times in one query (gate2/regenerate in a
loop), and keeping the raw sequence preserves that shape for later analysis
without a schema migration.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from verirag.config import settings

logger = logging.getLogger(__name__)

Outcome = Literal["verified", "abstained_insufficient", "abstained_ungrounded"]

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS query_metrics (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                TEXT    NOT NULL,
    query                    TEXT    NOT NULL,
    answer_format            TEXT    NOT NULL,
    corpus_only              INTEGER NOT NULL,
    outcome                  TEXT    NOT NULL,
    regeneration_count       INTEGER NOT NULL,
    external_fetch_attempted INTEGER NOT NULL,
    llm_call_count           INTEGER NOT NULL,
    total_latency_ms         REAL    NOT NULL,
    node_timings_json        TEXT    NOT NULL,
    estimated_cost_usd       REAL    NOT NULL
)
"""


@dataclass
class QueryMetricsRecord:
    """One query's measured outcome, mirroring the table columns."""

    timestamp: str
    query: str
    answer_format: str
    corpus_only: bool
    outcome: Outcome
    regeneration_count: int
    external_fetch_attempted: bool
    llm_call_count: int
    total_latency_ms: float
    node_timings: list[tuple[str, float]] = field(default_factory=list)
    estimated_cost_usd: float = 0.0


@dataclass
class MetricsSummary:
    """Aggregate view over all logged queries."""

    total_queries: int
    abstention_rate: float
    # Fraction of queries where the corrective loop engaged at all:
    # a regeneration happened, or an external fetch was attempted.
    loop_trigger_rate: float
    p95_latency_by_node: dict[str, float]
    avg_cost_per_query_usd: float


def _resolve_path(db_path: str | None) -> Path:
    return Path(db_path or settings.metrics_db_path)


def init_db(db_path: str | None = None) -> None:
    """Create the metrics table if it does not already exist."""
    path = _resolve_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(_TABLE_DDL)


def log_query_metrics(
    record: QueryMetricsRecord, db_path: str | None = None
) -> None:
    """Insert one metrics row, creating the table on first use."""
    path = _resolve_path(db_path)
    init_db(str(path))
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO query_metrics (
                timestamp, query, answer_format, corpus_only, outcome,
                regeneration_count, external_fetch_attempted, llm_call_count,
                total_latency_ms, node_timings_json, estimated_cost_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.timestamp,
                record.query,
                record.answer_format,
                int(record.corpus_only),
                record.outcome,
                record.regeneration_count,
                int(record.external_fetch_attempted),
                record.llm_call_count,
                float(record.total_latency_ms),
                json.dumps(record.node_timings),
                float(record.estimated_cost_usd),
            ),
        )


def summarize_metrics(db_path: str | None = None) -> MetricsSummary:
    """Aggregate every logged query into a MetricsSummary."""
    path = _resolve_path(db_path)
    if not path.exists():
        return MetricsSummary(0, 0.0, 0.0, {}, 0.0)

    init_db(str(path))
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """
            SELECT outcome, regeneration_count, external_fetch_attempted,
                   node_timings_json, estimated_cost_usd
            FROM query_metrics
            """
        ).fetchall()

    total = len(rows)
    if total == 0:
        return MetricsSummary(0, 0.0, 0.0, {}, 0.0)

    abstained = sum(1 for row in rows if str(row[0]).startswith("abstained"))
    looped = sum(1 for row in rows if row[1] > 0 or row[2])
    total_cost = sum(float(row[4]) for row in rows)

    # Group every node's elapsed times across ALL rows, then take p95 per node.
    by_node: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        try:
            timings = json.loads(row[3]) or []
        except (json.JSONDecodeError, TypeError):
            logger.warning("Skipping unparseable node_timings_json row.")
            continue
        for entry in timings:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            node_name, elapsed_ms = entry
            by_node[str(node_name)].append(float(elapsed_ms))

    p95 = {
        node_name: float(np.percentile(values, 95))
        for node_name, values in sorted(by_node.items())
        if values
    }

    return MetricsSummary(
        total_queries=total,
        abstention_rate=abstained / total,
        loop_trigger_rate=looped / total,
        p95_latency_by_node=p95,
        avg_cost_per_query_usd=total_cost / total,
    )
