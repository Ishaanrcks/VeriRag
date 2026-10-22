"""Graph nodes for the VeriRAG query pipeline.

Each node takes the state and returns only the keys it changes. Counters are
returned as new absolute values (replace semantics); ``trace`` entries are
returned as a one-element list and concatenated by the state's reducer.

Only DeepSeek calls count toward ``llm_call_count``: Gate 1 counts one only
when it escalates, Gate 2 counts one for claim extraction (its NLI scoring is
a local encoder, not an LLM), and generation/regeneration count one each.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable

from verirag.config import settings
from verirag.critic.gate1_coverage import check_coverage
from verirag.critic.gate2_groundedness import check_groundedness
from verirag.external.fetcher import fetch_external
from verirag.generation.generator import generate_answer
from verirag.graph.state import VeriRAGState
from verirag.ingest import ingest_documents
from verirag.retrieval.hybrid import hybrid_retrieve

logger = logging.getLogger(__name__)


def _timed(
    node_name: str,
) -> Callable[[Callable[[VeriRAGState], VeriRAGState]], Callable[
    [VeriRAGState], VeriRAGState
]]:
    """Time a node and attach ``(node_name, elapsed_ms)`` to its update.

    Applied as a decorator so the timer wraps each node body exactly once,
    rather than being repeated (and eventually forgotten) in nine places. The
    state's reducer concatenates the entries, so repeat visits accumulate.
    """

    def decorator(
        func: Callable[[VeriRAGState], VeriRAGState],
    ) -> Callable[[VeriRAGState], VeriRAGState]:
        @functools.wraps(func)
        def wrapper(state: VeriRAGState) -> VeriRAGState:
            start = time.perf_counter()
            update = dict(func(state))
            update["node_timings"] = [
                (node_name, (time.perf_counter() - start) * 1000.0)
            ]
            return update

        return wrapper

    return decorator


@_timed("retrieve")
def retrieve_node(state: VeriRAGState) -> VeriRAGState:
    """Hybrid-retrieve for the query (dense + sparse -> RRF -> rerank)."""
    hits = hybrid_retrieve(state["query"])
    top = hits[0].rerank_score if hits else None
    return {
        "hits": hits,
        "trace": [
            f"retrieve: {len(hits)} hits"
            + (f", top rerank_score={top:.6f}" if top is not None else "")
        ],
    }


@_timed("gate1")
def gate1_node(state: VeriRAGState) -> VeriRAGState:
    """Gate 1: does the retrieved evidence cover the query at all?"""
    verdict = check_coverage(state["query"], state.get("hits", []))

    llm_calls = state.get("llm_call_count", 0)
    if verdict.decision_path == "llm_escalation":
        llm_calls += 1

    trace = [
        f"gate1: verdict={verdict.verdict} via {verdict.decision_path} "
        f"(max_rerank_score={verdict.max_rerank_score:.6f}, "
        f"score_spread={verdict.score_spread:.6f})"
    ]
    if verdict.llm_reason:
        trace.append(f"gate1: judge reason: {verdict.llm_reason}")

    # Monotonic-improvement observation after an external fetch. The
    # fetch-once bound already prevents further looping, so this is a logged
    # signal rather than a control branch.
    pre_fetch = state.get("pre_fetch_max_score")
    if pre_fetch is not None and verdict.max_rerank_score <= pre_fetch:
        trace.append(
            "external fetch did not improve coverage "
            f"(max_rerank_score {pre_fetch:.6f} -> "
            f"{verdict.max_rerank_score:.6f})"
        )

    return {"gate1_verdict": verdict, "llm_call_count": llm_calls, "trace": trace}


@_timed("fetch_external")
def fetch_external_node(state: VeriRAGState) -> VeriRAGState:
    """Fetch external material once and make it searchable."""
    verdict = state.get("gate1_verdict")
    pre_fetch_max_score = verdict.max_rerank_score if verdict else 0.0

    documents = fetch_external(state["query"])
    trace = [f"fetch_external: returned {len(documents)} document(s)"]

    if documents:
        # Push fetched material through the normal ingestion pipeline so it is
        # chunked, contextualized, embedded and stored like any local file.
        # No-op with the Phase 8 stub, but wired for Phase 9.
        results = ingest_documents(documents)
        ingested_chunks = sum(
            r.num_text_chunks + r.num_figure_chunks for r in results
        )
        trace.append(
            f"fetch_external: ingested {len(results)} source(s), "
            f"{ingested_chunks} chunk(s)"
        )

    return {
        "external_fetch_attempted": True,
        "pre_fetch_max_score": pre_fetch_max_score,
        "trace": trace,
    }


@_timed("generate")
def generate_node(state: VeriRAGState) -> VeriRAGState:
    """Generate a cited answer from the retrieved hits."""
    answer = generate_answer(
        state["query"], state.get("hits", []), state["answer_format"]
    )
    return {
        "generated_answer": answer,
        "llm_call_count": state.get("llm_call_count", 0) + 1,
        "trace": [
            f"generate: {len(answer.answer_text)} chars, "
            f"{len(answer.citations)} citation(s), "
            f"format={state['answer_format'].value}"
        ],
    }


@_timed("gate2")
def gate2_node(state: VeriRAGState) -> VeriRAGState:
    """Gate 2: is every factual claim entailed by its cited evidence?"""
    answer = state["generated_answer"]
    verdict = check_groundedness(answer)
    return {
        "gate2_verdict": verdict,
        # Claim extraction is one DeepSeek call; NLI scoring is local.
        "llm_call_count": state.get("llm_call_count", 0) + 1,
        "trace": [
            f"gate2: overall_grounded={verdict.overall_grounded}, "
            f"{len(verdict.claims)} claim(s), "
            f"{len(verdict.unsupported_claims)} unsupported"
        ],
    }


@_timed("regenerate")
def regenerate_node(state: VeriRAGState) -> VeriRAGState:
    """Regenerate, explicitly steering away from the unverified claims."""
    verdict = state["gate2_verdict"]
    avoid = [claim.claim_text for claim in verdict.unsupported_claims]

    answer = generate_answer(
        state["query"],
        state.get("hits", []),
        state["answer_format"],
        avoid_claims=avoid,
    )
    attempt = state.get("regeneration_count", 0) + 1
    return {
        "generated_answer": answer,
        "regeneration_count": attempt,
        "llm_call_count": state.get("llm_call_count", 0) + 1,
        "trace": [
            f"regenerate (attempt {attempt}/{settings.max_critic_iterations}): "
            f"avoiding {len(avoid)} unverified claim(s)"
        ],
    }


@_timed("finalize_verified")
def finalize_verified_node(state: VeriRAGState) -> VeriRAGState:
    """Accept the verified answer as final."""
    answer = state["generated_answer"]
    return {
        "final_answer": answer.answer_text,
        "final_citations": answer.citations,
        "abstained": False,
        "trace": [
            "finalize_verified: answer passed both gates "
            f"(regenerations={state.get('regeneration_count', 0)})"
        ],
    }


@_timed("abstain_insufficient_coverage")
def abstain_insufficient_coverage_node(state: VeriRAGState) -> VeriRAGState:
    """Abstain because the corpus does not cover the question."""
    verdict = state.get("gate1_verdict")

    if verdict and verdict.decision_path == "llm_escalation" and verdict.llm_reason:
        reason = (
            "Insufficient coverage to answer this question. "
            f"{verdict.llm_reason}"
        )
    else:
        reason = (
            "Insufficient coverage to answer this question: the available "
            "documents do not contain information relevant to this question."
        )
    if state.get("corpus_only"):
        reason += " External sources were not consulted (corpus-only mode)."
    elif state.get("external_fetch_attempted"):
        # Distinguish "the search found nothing" from "it found material that
        # still did not answer the question" -- claiming the former when
        # coverage measurably improved would be misleading.
        pre_fetch = state.get("pre_fetch_max_score")
        post_fetch = verdict.max_rerank_score if verdict else 0.0
        if pre_fetch is not None and post_fetch > pre_fetch:
            reason += (
                " An external search was attempted; the material it found did "
                "not answer the question."
            )
        else:
            reason += (
                " An external search was attempted and added no new material."
            )

    return {
        "abstained": True,
        "abstention_reason": reason,
        "final_answer": None,
        "final_citations": [],
        "trace": ["abstain_insufficient_coverage: refused to answer"],
    }


@_timed("abstain_ungrounded")
def abstain_ungrounded_node(state: VeriRAGState) -> VeriRAGState:
    """Abstain because claims stayed unverified after exhausting retries."""
    verdict = state.get("gate2_verdict")
    unsupported = (
        [claim.claim_text for claim in verdict.unsupported_claims] if verdict else []
    )

    reason = (
        "Could not produce a fully grounded answer after "
        f"{state.get('regeneration_count', 0)} regeneration attempt(s). "
        "These claims could not be verified against the sources:"
    )
    for claim_text in unsupported:
        reason += f"\n- {claim_text}"

    return {
        "abstained": True,
        "abstention_reason": reason,
        "final_answer": None,
        "final_citations": [],
        "trace": [
            "abstain_ungrounded: refused to return an unverified answer "
            f"({len(unsupported)} unsupported claim(s))"
        ],
    }
