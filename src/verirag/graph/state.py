"""LangGraph state for the VeriRAG query pipeline.

``trace`` is annotated with ``operator.add`` so each node returns only the
lines it adds and LangGraph concatenates them, giving an ordered record of
every step taken. All other fields use the default replace semantics, so
counters are returned as new absolute values rather than deltas.

The state is ``total=False`` because nodes return partial updates.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from verirag.critic.gate1_coverage import Gate1Verdict
from verirag.critic.gate2_groundedness import Gate2Verdict
from verirag.generation.generator import AnswerFormat, Citation, GeneratedAnswer
from verirag.pipeline_log import LogEvent
from verirag.retrieval.types import SearchHit


class VeriRAGState(TypedDict, total=False):
    """Everything one query carries through the graph."""

    # --- inputs ---
    query: str
    answer_format: AnswerFormat
    # When True, the external-fetch branch is blocked entirely.
    corpus_only: bool
    # Multi-tenant isolation: retrieval and document listing are scoped to
    # shared/demo content plus this session's own uploads. None -> shared only.
    session_id: str | None

    # --- document-reference resolution (runs before retrieval) ---
    # doc_ids explicitly named in the query, and whether the query reads as
    # "summarize this" rather than a content question.
    resolved_doc_ids: list[str]
    is_summarization_intent: bool
    # The query with any named filenames stripped out, used for dense/sparse
    # search so a bare filename does not pollute the embedding.
    search_query: str | None

    # --- retrieval ---
    hits: list[SearchHit]
    # Freshly fetched external chunks, scoped to THIS query only. Retrieval
    # always excludes persisted external content, so a legitimate fallback
    # reaches generation through here rather than through the store.
    external_hits: list[SearchHit]

    # --- Gate 1 ---
    gate1_verdict: Gate1Verdict | None
    external_fetch_attempted: bool
    # max_rerank_score observed before fetching, for the monotonic-improvement
    # observation after re-retrieval.
    pre_fetch_max_score: float | None

    # --- generation + Gate 2 ---
    generated_answer: GeneratedAnswer | None
    gate2_verdict: Gate2Verdict | None

    # --- bounds ---
    regeneration_count: int
    llm_call_count: int

    # --- outputs ---
    final_answer: str | None
    final_citations: list[Citation]
    abstained: bool
    abstention_reason: str | None

    # --- observability ---
    trace: Annotated[list[str], operator.add]
    # Structured per-stage events for the UI's Pipeline Log tab. Accumulates
    # like ``trace``; purely additive -- no node reads it.
    pipeline_log: Annotated[list[LogEvent], operator.add]
    # (node_name, elapsed_ms) per node visit, accumulated the same way as
    # trace. A node visited twice (gate2/regenerate in a loop) contributes two
    # entries under the same name, which is what we want for the latency
    # distribution.
    node_timings: Annotated[list[tuple[str, float]], operator.add]
