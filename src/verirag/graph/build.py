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
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph

from verirag.config import settings
from verirag.critic.gate1_coverage import Gate1Verdict
from verirag.critic.gate2_groundedness import Gate2Verdict
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
    finalize_partial_node,
    has_verified_claim,
    resolve_intent_node,
    retrieve_node,
    summarize_document_node,
)
from verirag.graph.state import VeriRAGState
from verirag.observability.metrics import QueryMetricsRecord, log_query_metrics
from verirag.pipeline_log import LogEvent
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

    # Presentation-only. These are the SAME values already computed for the
    # metrics row, surfaced so the UI need not recompute or re-derive them.
    # "verified" | "partial" | "abstained_insufficient" | "abstained_ungrounded".
    # "partial": some claims verified, some not; the answer is returned with an
    # explicit list of the verified ones (verified_facts).
    outcome: str = "verified"
    total_latency_ms: float = 0.0
    llm_call_count: int = 0
    estimated_cost_usd: float = 0.0
    # BENCHMARK FLAG unverified_candidate. Strictly separate from `answer`: the
    # candidate is what the pipeline WOULD have said with nothing verifying it,
    # measured at 50% precision, so it is a lead and never an answer. None unless
    # the flag is on and the pipeline declined.
    candidate_answer: str | None = None
    candidate_reason: str | None = None
    candidate_sources: list = field(default_factory=list)
    gate1_verdict: Gate1Verdict | None = None
    gate2_verdict: Gate2Verdict | None = None
    # Citations from the last generation attempt. On ``verified`` this
    # duplicates ``citations``; on ``abstained_ungrounded`` it carries the
    # citations the generator SAW (``final_citations`` is cleared on that
    # path), so the UI can surface raw retrieved excerpts to the user instead
    # of leaving them with nothing. Presentation-only -- zero effect on
    # scoring, retrieval, or critic behavior.
    retrieved_citations: list[Citation] = field(default_factory=list)
    # Structured per-stage events (retrieval, context, gates, regeneration,
    # outcome) for the UI's Pipeline Log tab. Presentation-only.
    pipeline_log: list[LogEvent] = field(default_factory=list)
    # For outcome == "partial": the claims that passed Gate 2 (VerifiedFact).
    verified_facts: list = field(default_factory=list)


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
    """Finalize, regenerate within bounds, answer partially, or abstain.

    Once regeneration is over, a final check with at least one verified claim
    answers (``finalize_partial``) and one with none abstains.
    """
    verdict = state.get("gate2_verdict")
    if verdict is not None and verdict.overall_grounded:
        return "finalize_verified"

    # No regeneration on the summarization path: regenerate_node runs the
    # question-answering generator, which would silently replace a summary
    # with a Q&A answer. It goes straight to the end-of-loop decision below.
    if not state.get("is_summarization_intent") and (
        state.get("regeneration_count", 0) < settings.max_critic_iterations
        and state.get("llm_call_count", 0) < settings.max_llm_calls_per_query
    ):
        return "regenerate"
    return "finalize_partial" if has_verified_claim(state) else "abstain_ungrounded"


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
    graph.add_node("finalize_partial", finalize_partial_node)

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
            "finalize_partial": "finalize_partial",
            "abstain_ungrounded": "abstain_ungrounded",
        },
    )
    graph.add_edge("regenerate", "gate2")

    graph.add_edge("finalize_verified", END)
    graph.add_edge("finalize_partial", END)
    graph.add_edge("abstain_insufficient_coverage", END)
    graph.add_edge("abstain_ungrounded", END)

    return graph


# Compiled once at module load and reused for every query.
COMPILED_GRAPH = _build_graph().compile()


def _initial_state(
    query: str,
    answer_format: AnswerFormat,
    corpus_only: bool,
    session_id: str | None = None,
) -> VeriRAGState:
    return {
        "query": query,
        "answer_format": answer_format,
        "corpus_only": corpus_only,
        "session_id": session_id,
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
        "answer_is_partial": False,
        "verified_facts": [],
        "trace": [],
        "node_timings": [],
        "pipeline_log": [],
        "resolved_doc_ids": [],
        "is_summarization_intent": False,
        "search_query": None,
    }


def _derive_outcome(final_state: VeriRAGState) -> str:
    """Classify how the query ended, from the state rather than message text."""
    if not final_state.get("abstained"):
        return "partial" if final_state.get("answer_is_partial") else "verified"
    # Coverage abstention happens before generation, so gate2 never ran.
    gate2_verdict = final_state.get("gate2_verdict")
    if gate2_verdict is not None and not gate2_verdict.overall_grounded:
        return "abstained_ungrounded"
    return "abstained_insufficient"


def _derived_values(final_state: VeriRAGState) -> dict:
    """Values derived from the final state, computed once.

    Both the metrics row and VeriRAGResult read this, so the number the UI
    shows and the number logged to SQLite cannot drift apart.
    """
    node_timings = [
        (str(name), float(elapsed))
        for name, elapsed in (final_state.get("node_timings") or [])
    ]
    llm_call_count = final_state.get("llm_call_count", 0)
    return dict(
        node_timings=node_timings,
        total_latency_ms=sum(elapsed for _, elapsed in node_timings),
        llm_call_count=llm_call_count,
        estimated_cost_usd=(
            llm_call_count * settings.estimated_cost_per_llm_call_usd
        ),
        outcome=_derive_outcome(final_state),
    )


def _log_metrics(
    query: str,
    answer_format: AnswerFormat,
    corpus_only: bool,
    final_state: VeriRAGState,
) -> None:
    """Record one query's metrics. Never raises -- metrics are not the product."""
    try:
        derived = _derived_values(final_state)
        log_query_metrics(
            QueryMetricsRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                query=query,
                answer_format=answer_format.value,
                corpus_only=corpus_only,
                outcome=derived["outcome"],  # type: ignore[arg-type]
                regeneration_count=final_state.get("regeneration_count", 0),
                external_fetch_attempted=bool(
                    final_state.get("external_fetch_attempted")
                ),
                llm_call_count=derived["llm_call_count"],
                total_latency_ms=derived["total_latency_ms"],
                node_timings=derived["node_timings"],
                estimated_cost_usd=derived["estimated_cost_usd"],
            )
        )
    except Exception as exc:  # noqa: BLE001 - never break a user-facing query
        logger.warning(
            "Failed to log query metrics (%s): %s", type(exc).__name__, exc
        )


# Fields whose updates ACCUMULATE across node visits (operator.add in
# VeriRAGState). Every other field replaces on update, so merging stream
# updates into one final_state has to respect this difference.
_ACCUMULATING_STATE_KEYS = ("trace", "node_timings", "pipeline_log")


def _merge_update(
    final_state: VeriRAGState, node_update: dict
) -> list[str]:
    """Apply one node's update to ``final_state`` in place; return its trace lines."""
    emitted: list[str] = []
    for key, value in node_update.items():
        if key in _ACCUMULATING_STATE_KEYS:
            current = list(final_state.get(key) or [])  # type: ignore[arg-type]
            current.extend(value or [])
            final_state[key] = current  # type: ignore[literal-required]
            if key == "trace":
                emitted = list(value or [])
        else:
            final_state[key] = value  # type: ignore[literal-required]
    return emitted


def _build_result(
    final_state: VeriRAGState,
    query: str,
    answer_format: AnswerFormat,
    corpus_only: bool,
) -> VeriRAGResult:
    """Shared VeriRAGResult assembly for both streaming and blocking entry points."""
    _log_metrics(query, answer_format, corpus_only, final_state)
    derived = _derived_values(final_state)
    generated_answer = final_state.get("generated_answer")
    retrieved_citations = (
        list(generated_answer.citations) if generated_answer else []
    )
    candidate = _unverified_candidate(final_state, derived["outcome"])
    return VeriRAGResult(
        answer=final_state.get("final_answer"),
        citations=final_state.get("final_citations") or [],
        abstained=bool(final_state.get("abstained")),
        abstention_reason=final_state.get("abstention_reason"),
        trace=final_state.get("trace") or [],
        regeneration_count=final_state.get("regeneration_count", 0),
        outcome=derived["outcome"],
        total_latency_ms=derived["total_latency_ms"],
        llm_call_count=derived["llm_call_count"],
        estimated_cost_usd=derived["estimated_cost_usd"],
        gate1_verdict=final_state.get("gate1_verdict"),
        gate2_verdict=final_state.get("gate2_verdict"),
        retrieved_citations=retrieved_citations,
        pipeline_log=list(final_state.get("pipeline_log") or []),
        verified_facts=list(final_state.get("verified_facts") or []),
        candidate_answer=candidate.answer if candidate else None,
        candidate_reason=candidate.reason if candidate else None,
        candidate_sources=candidate.sources if candidate else [],
    )


def _unverified_candidate(final_state: dict, outcome: str):
    """The ungated answer for a declined question, or None.

    Produced only under the ``unverified_candidate`` flag, only when the pipeline
    did NOT return a verified answer, and only when generation actually ran. A Gate 1
    decline means nothing relevant was retrieved, so there is nothing to offer and
    none is made -- the rule exists so that declining for want of evidence can never
    be turned into a guess.
    """
    from verirag.config import settings

    if not settings.unverified_candidate:
        return None
    if outcome == "verified":
        return None
    if outcome == "abstained_insufficient":   # Gate 1 declined: no evidence, no candidate
        return None
    generated = final_state.get("generated_answer")
    if generated is None or not (getattr(generated, "answer_text", "") or "").strip():
        return None
    from verirag.generation.candidate import build_candidate

    return build_candidate(generated.answer_text, final_state.get("gate2_verdict"),
                           list(getattr(generated, "hits_used", None) or []))


def run_query_streaming(
    query: str,
    answer_format: AnswerFormat,
    corpus_only: bool = False,
    session_id: str | None = None,
) -> Iterator[tuple[str, str]]:
    """Stream (node_name, trace_summary) tuples as each pipeline node completes.

    The generator ``return``s the final ``VeriRAGResult`` (via StopIteration.value),
    so callers can iterate for live updates and still get the final result at the
    end -- no second traversal, no duplicated node runs. Trace strings are the
    same ones each node writes into ``state['trace']``, so the live view and the
    collapsed post-hoc expander share one source of truth and cannot drift.

    ``session_id`` scopes retrieval/listing to shared content plus this
    session's uploads (multi-tenant isolation for the public deploy).
    """
    final_state: VeriRAGState = {}  # type: ignore[typeddict-item]
    for update in COMPILED_GRAPH.stream(
        _initial_state(query, answer_format, corpus_only, session_id),
        stream_mode="updates",
    ):
        for node_name, node_update in update.items():
            if not node_update:
                continue
            for line in _merge_update(final_state, node_update):
                yield node_name, line

    return _build_result(final_state, query, answer_format, corpus_only)


def run_query(
    query: str,
    answer_format: AnswerFormat,
    corpus_only: bool = False,
    session_id: str | None = None,
) -> VeriRAGResult:
    """Blocking convenience wrapper over ``run_query_streaming``.

    Exhausts the streaming generator to collect the final result. Preserves the
    original single-call API for anything (tests, eval harness, programmatic
    use) that doesn't need the per-step updates.
    """
    gen = run_query_streaming(query, answer_format, corpus_only, session_id)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value
