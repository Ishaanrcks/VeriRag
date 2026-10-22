"""LangGraph wiring for VeriRAG, plus the top-level ``run_query`` entry point.

Control flow:

    START -> resolve_intent
      summarize intent + exactly 1 named doc -> summarize_document -> gate2
      otherwise                              -> retrieve
    retrieve -> gate1
      gate1 sufficient                  -> generate
      gate1 insufficient + can fetch    -> fetch_external -> retrieve (once)
      gate1 insufficient + cannot fetch -> abstain_insufficient_coverage
    generate -> gate2
      gate2 grounded                    -> finalize_verified
      gate2 ungrounded + retries left   -> regenerate -> gate2
      gate2 ungrounded + exhausted      -> abstain_ungrounded

Both loops are bounded. ``fetch_external`` re-enters via ``retrieve``, but
because it sets ``external_fetch_attempted=True`` the Gate 1 router can never
choose it twice -- this is the spec's "re-enter the loop once". The
regeneration loop is bounded by ``max_critic_iterations``, and both branches
additionally respect ``max_llm_calls_per_query``. The system therefore never
returns an unverified answer: it either verifies, or abstains.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph

from verirag.config import settings
from verirag.generation.generator import AnswerFormat, Citation
from verirag.graph.nodes import (
    abstain_insufficient_coverage_node,
    abstain_ungrounded_node,
    fetch_external_node,
    finalize_verified_node,
    gate1_node,
    gate2_node,
    generate_node,
    regenerate_node,
    resolve_intent_node,
    retrieve_node,
    summarize_document_node,
)
from verirag.graph.state import VeriRAGState
from verirag.observability.metrics import QueryMetricsRecord, log_query_metrics
from verirag.observability.tracing import init_tracing

logger = logging.getLogger(__name__)

# Activate tracing before any query can run. No-op without a real API key.
init_tracing()


@dataclass
class VeriRAGResult:
    """The outcome of one query: a verified answer, or an abstention."""

    answer: str | None
    citations: list[Citation] = field(default_factory=list)
    abstained: bool = False
    abstention_reason: str | None = None
    trace: list[str] = field(default_factory=list)
    regeneration_count: int = 0


def _route_after_resolve_intent(state: VeriRAGState) -> str:
    """Summarize a named document, or fall through to normal retrieval.

    Whole-document summarization requires BOTH summarization intent and
    exactly one resolved document. Two named documents is a comparison
    question, not a summarization one, so it takes the retrieval path.
    """
    resolved = state.get("resolved_doc_ids") or []
    if state.get("is_summarization_intent") and len(resolved) == 1:
        return "summarize_document"
    return "retrieve"


def _route_after_gate1(state: VeriRAGState) -> str:
    """Generate, fetch externally once, or abstain."""
    verdict = state.get("gate1_verdict")
    if verdict is not None and verdict.verdict == "sufficient":
        return "generate"

    # Insufficient coverage: may we try an external fetch?
    if (
        state.get("corpus_only")
        or state.get("external_fetch_attempted")
        or state.get("llm_call_count", 0) >= settings.max_llm_calls_per_query
    ):
        return "abstain_insufficient_coverage"
    return "fetch_external"


def _route_after_gate2(state: VeriRAGState) -> str:
    """Finalize, regenerate within bounds, or abstain."""
    verdict = state.get("gate2_verdict")
    if verdict is not None and verdict.overall_grounded:
        return "finalize_verified"

    # No regeneration on the summarization path: regenerate_node runs the
    # question-answering generator, which would silently replace a summary
    # with a Q&A answer. An ungrounded summary abstains instead, preserving
    # the "never return unverified" invariant.
    if state.get("is_summarization_intent"):
        return "abstain_ungrounded"

    if (
        state.get("regeneration_count", 0) < settings.max_critic_iterations
        and state.get("llm_call_count", 0) < settings.max_llm_calls_per_query
    ):
        return "regenerate"
    return "abstain_ungrounded"


def _build_graph() -> StateGraph:
    graph = StateGraph(VeriRAGState)

    graph.add_node("resolve_intent", resolve_intent_node)
    graph.add_node("summarize_document", summarize_document_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("gate1", gate1_node)
    graph.add_node("fetch_external", fetch_external_node)
    graph.add_node("generate", generate_node)
    graph.add_node("gate2", gate2_node)
    graph.add_node("regenerate", regenerate_node)
    graph.add_node("finalize_verified", finalize_verified_node)
    graph.add_node(
        "abstain_insufficient_coverage", abstain_insufficient_coverage_node
    )
    graph.add_node("abstain_ungrounded", abstain_ungrounded_node)

    graph.add_edge(START, "resolve_intent")
    graph.add_conditional_edges(
        "resolve_intent",
        _route_after_resolve_intent,
        {
            "summarize_document": "summarize_document",
            "retrieve": "retrieve",
        },
    )
    # Summarization skips Gate 1 (coverage is self-evident) but NOT Gate 2.
    graph.add_edge("summarize_document", "gate2")

    graph.add_edge("retrieve", "gate1")

    graph.add_conditional_edges(
        "gate1",
        _route_after_gate1,
        {
            "generate": "generate",
            "fetch_external": "fetch_external",
            "abstain_insufficient_coverage": "abstain_insufficient_coverage",
        },
    )

    # Re-retrieve so any newly ingested material is picked up; the
    # external_fetch_attempted flag stops a second fetch.
    graph.add_edge("fetch_external", "retrieve")

    graph.add_edge("generate", "gate2")
    graph.add_conditional_edges(
        "gate2",
        _route_after_gate2,
        {
            "finalize_verified": "finalize_verified",
            "regenerate": "regenerate",
            "abstain_ungrounded": "abstain_ungrounded",
        },
    )
    graph.add_edge("regenerate", "gate2")

    graph.add_edge("finalize_verified", END)
    graph.add_edge("abstain_insufficient_coverage", END)
    graph.add_edge("abstain_ungrounded", END)

    return graph


# Compiled once at module load and reused for every query.
COMPILED_GRAPH = _build_graph().compile()


def _initial_state(
    query: str, answer_format: AnswerFormat, corpus_only: bool
) -> VeriRAGState:
    return {
        "query": query,
        "answer_format": answer_format,
        "corpus_only": corpus_only,
        "hits": [],
        "external_hits": [],
        "gate1_verdict": None,
        "external_fetch_attempted": False,
        "pre_fetch_max_score": None,
        "generated_answer": None,
        "gate2_verdict": None,
        "regeneration_count": 0,
        "llm_call_count": 0,
        "final_answer": None,
        "final_citations": [],
        "abstained": False,
        "abstention_reason": None,
        "trace": [],
        "node_timings": [],
        "resolved_doc_ids": [],
        "is_summarization_intent": False,
        "search_query": None,
    }


def _derive_outcome(final_state: VeriRAGState) -> str:
    """Classify how the query ended, from the state rather than message text."""
    if not final_state.get("abstained"):
        return "verified"
    # Coverage abstention happens before generation, so gate2 never ran.
    gate2_verdict = final_state.get("gate2_verdict")
    if gate2_verdict is not None and not gate2_verdict.overall_grounded:
        return "abstained_ungrounded"
    return "abstained_insufficient"


def _log_metrics(
    query: str,
    answer_format: AnswerFormat,
    corpus_only: bool,
    final_state: VeriRAGState,
) -> None:
    """Record one query's metrics. Never raises -- metrics are not the product."""
    try:
        node_timings = [
            (str(name), float(elapsed))
            for name, elapsed in (final_state.get("node_timings") or [])
        ]
        llm_call_count = final_state.get("llm_call_count", 0)
        log_query_metrics(
            QueryMetricsRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                query=query,
                answer_format=answer_format.value,
                corpus_only=corpus_only,
                outcome=_derive_outcome(final_state),  # type: ignore[arg-type]
                regeneration_count=final_state.get("regeneration_count", 0),
                external_fetch_attempted=bool(
                    final_state.get("external_fetch_attempted")
                ),
                llm_call_count=llm_call_count,
                total_latency_ms=sum(elapsed for _, elapsed in node_timings),
                node_timings=node_timings,
                estimated_cost_usd=(
                    llm_call_count * settings.estimated_cost_per_llm_call_usd
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001 - never break a user-facing query
        logger.warning(
            "Failed to log query metrics (%s): %s", type(exc).__name__, exc
        )


def run_query(
    query: str,
    answer_format: AnswerFormat,
    corpus_only: bool = False,
) -> VeriRAGResult:
    """Answer a query end to end. The single entry point for the Phase 11 UI."""
    final_state = COMPILED_GRAPH.invoke(
        _initial_state(query, answer_format, corpus_only)
    )
    _log_metrics(query, answer_format, corpus_only, final_state)
    return VeriRAGResult(
        answer=final_state.get("final_answer"),
        citations=final_state.get("final_citations") or [],
        abstained=bool(final_state.get("abstained")),
        abstention_reason=final_state.get("abstention_reason"),
        trace=final_state.get("trace") or [],
        regeneration_count=final_state.get("regeneration_count", 0),
    )
