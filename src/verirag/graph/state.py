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
from verirag.retrieval.types import SearchHit


class VeriRAGState(TypedDict, total=False):
    """Everything one query carries through the graph."""

    # --- inputs ---
    query: str
    answer_format: AnswerFormat
    # When True, the external-fetch branch is blocked entirely.
    corpus_only: bool

    # --- retrieval ---
    hits: list[SearchHit]

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
