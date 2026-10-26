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
from verirag.generation.generator import (
    GeneratedAnswer,
    build_context_block,
    generate_answer,
)
from verirag.graph.state import VeriRAGState
from verirag.ingest import ingest_documents
from verirag.llm.client import chat_completion
from verirag.pipeline_log import (
    abstain_event,
    answer_event,
    context_event,
    gate1_event,
    gate2_event,
    regen_event,
    retrieval_event,
)
from verirag.retrieval.doc_resolver import (
    detect_summarization_intent,
    resolve_document_references,
    strip_document_references,
)
from verirag.retrieval.hybrid import hybrid_retrieve, score_external_hits
from verirag.retrieval.types import SearchHit
from verirag.retrieval.vector_store import (
    get_chunks_by_doc_id,
    list_ingested_documents,
)

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


_SUMMARY_SYSTEM_PROMPT = """\
You are summarizing a single source document for a reader who has not read it.

Rules you must follow:
- Summarize ONLY what the document actually says. Do not add outside
  knowledge, and do not infer conclusions the document does not state.
- Every statement you make comes from this one document, which is numbered
  [1]. End each sentence or bullet with [1] so each claim is attributable.
- Be faithful over fluent: if the document is unclear on something, say so
  rather than smoothing it over."""

_SUMMARY_SINGLE_PASS = """\
Summarize the document below.

<document>
{document_text}
</document>

Write a structured summary: what the document is about, its main
contributions or points, and any key findings or conclusions. Use short
headings or bullets. Remember to end each sentence or bullet with [1]."""

_SUMMARY_MAP = """\
Below is one consecutive section of a longer document.

<section>
{section_text}
</section>

Write a dense, factual digest of this section's content. Capture specifics
(definitions, findings, numbers, names) rather than writing generalities. Do
not add citation markers; this digest is an intermediate note."""

_SUMMARY_REDUCE = """\
Below are ordered digests of consecutive sections of one document.

<digests>
{digest_text}
</digests>

Write a single structured summary of the whole document from these digests:
what it is about, its main contributions or points, and key findings. Use
short headings or bullets. Remember to end each sentence or bullet with [1]."""

_SUMMARY_FAILED_MESSAGE = (
    "Could not summarize this document - the language model was unreachable."
)


def _ordered_batches(chunks: list[SearchHit], max_chars: int) -> list[str]:
    """Group chunks, in reading order, into batches under a character budget."""
    batches: list[str] = []
    current: list[str] = []
    current_len = 0
    for chunk in chunks:
        text = chunk.raw_text or ""
        if current and current_len + len(text) > max_chars:
            batches.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(text)
        current_len += len(text)
    if current:
        batches.append("\n\n".join(current))
    return batches


@_timed("resolve_intent")
def resolve_intent_node(state: VeriRAGState) -> VeriRAGState:
    """Detect named documents and summarization intent before retrieving.

    Runs before retrieval because "read this whole document" and "search for
    content matching this question" are different operations, and a filename
    plus an instruction makes a terrible search query.
    """
    query = state["query"]
    try:
        # Scope to shared content + this session's own uploads, so a user can
        # only name documents they are allowed to see.
        ingested = list_ingested_documents(session_id=state.get("session_id"))
    except Exception as exc:  # noqa: BLE001 - never block a query on this
        logger.warning("Could not list ingested documents (%s): %s",
                       type(exc).__name__, exc)
        ingested = []

    resolved = resolve_document_references(query, ingested)
    is_summarization = detect_summarization_intent(query)

    # Strip the filename out of the text used for search, so it does not
    # pollute the query embedding. Document scoping is handled by the filter.
    search_query = (
        strip_document_references(query, ingested) if resolved else query
    )

    trace = [
        f"resolve_intent: {len(resolved)} named document(s), "
        f"summarization_intent={is_summarization}"
    ]
    if resolved:
        trace.append(f"resolve_intent: resolved doc_ids={resolved}")
        if search_query != query:
            trace.append(f"resolve_intent: search query -> {search_query!r}")

    return {
        "resolved_doc_ids": resolved,
        "is_summarization_intent": is_summarization,
        "search_query": search_query,
        "trace": trace,
    }


@_timed("summarize_document")
def summarize_document_node(state: VeriRAGState) -> VeriRAGState:
    """Summarize one whole document, reading every chunk in order.

    Gate 1 is skipped on this path: when the request is "summarize this
    document", coverage is self-evident by construction. Gate 2 is NOT
    skipped -- the summary's claims are still checked against the actual
    source chunks, preserving the "never return unverified" invariant.
    """
    doc_ids = state.get("resolved_doc_ids") or []
    doc_id = doc_ids[0]
    chunks = get_chunks_by_doc_id(doc_id)

    trace = [
        f"summarize_document: doc_id={doc_id}, {len(chunks)} chunk(s) "
        "in reading order"
    ]
    llm_calls = state.get("llm_call_count", 0)

    if not chunks:
        logger.warning("No chunks found for doc_id=%s; cannot summarize.", doc_id)
        trace.append("summarize_document: no chunks found")
        return {
            "generated_answer": GeneratedAnswer(
                answer_text=(
                    "Could not summarize this document - no stored content was "
                    "found for it."
                ),
                citations=[],
                format_used=state["answer_format"],
                hits_used=[],
            ),
            "llm_call_count": llm_calls,
            "trace": trace,
        }

    # One citation per document: every chunk shares the same parent_text, so
    # build_context_block collapses them into [1] with all chunk_ids attached,
    # which is exactly the provenance Gate 2 needs.
    _, citations = build_context_block(chunks)
    ordered_text = "\n\n".join(chunk.raw_text for chunk in chunks)

    # Reserve one call for Gate 2's claim extraction.
    available_calls = max(
        0, settings.max_llm_calls_per_query - llm_calls - 1
    )
    partial_note = ""
    summary: str | None = None

    if available_calls < 1:
        logger.warning(
            "No LLM call budget left for summarization (used %d of %d).",
            llm_calls, settings.max_llm_calls_per_query,
        )
        trace.append("summarize_document: call budget exhausted before starting")
    elif len(ordered_text) <= settings.summarization_context_max_chars:
        trace.append(
            f"summarize_document: single pass ({len(ordered_text)} chars)"
        )
        summary = chat_completion(
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _SUMMARY_SINGLE_PASS.format(
                        document_text=ordered_text
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        llm_calls += 1
    else:
        batches = _ordered_batches(
            chunks, settings.summarization_batch_max_chars
        )
        # Each batch costs one call, plus one for the reduce step.
        if len(batches) + 1 > available_calls:
            affordable = max(1, available_calls - 1)
            logger.warning(
                "Summarization needs %d calls but only %d are available; "
                "summarizing the first %d of %d sections.",
                len(batches) + 1, available_calls, affordable, len(batches),
            )
            partial_note = (
                f"*(Partial summary: only the first {affordable} of "
                f"{len(batches)} sections of this document were processed "
                f"within the per-query call budget.)*\n\n"
            )
            batches = batches[:affordable]

        trace.append(
            f"summarize_document: map-reduce over {len(batches)} ordered "
            f"section(s) ({len(ordered_text)} chars total)"
        )

        digests: list[str] = []
        for i, batch in enumerate(batches, start=1):
            digest = chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": _SUMMARY_MAP.format(section_text=batch),
                    }
                ],
                temperature=0.2,
                max_tokens=700,
            )
            llm_calls += 1
            if digest:
                digests.append(f"[section {i}]\n{digest}")
            else:
                logger.warning("Section %d digest failed; skipping it.", i)

        if digests:
            summary = chat_completion(
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _SUMMARY_REDUCE.format(
                            digest_text="\n\n".join(digests)
                        ),
                    },
                ],
                temperature=0.2,
                max_tokens=1200,
            )
            llm_calls += 1
            trace.append(
                f"summarize_document: reduced {len(digests)} section digest(s)"
            )

    if summary is None:
        trace.append("summarize_document: summarization failed")
        answer_text = _SUMMARY_FAILED_MESSAGE
        used_citations: list = []
    else:
        answer_text = f"{partial_note}{summary.strip()}"
        used_citations = citations

    # Whole-document path: no ranking happens, so the "retrieval" is the
    # document's chunks in reading order, and the context is its full text.
    summary_mode = (
        "single_pass"
        if len(ordered_text) <= settings.summarization_context_max_chars
        else "map_reduce"
    )
    return {
        "generated_answer": GeneratedAnswer(
            answer_text=answer_text,
            citations=used_citations,
            format_used=state["answer_format"],
            hits_used=chunks,
        ),
        "llm_call_count": llm_calls,
        "pipeline_log": [
            retrieval_event(chunks, "Read"),
            context_event(chunks, ordered_text, mode=summary_mode),
        ],
        "trace": trace,
    }


@_timed("retrieve")
def retrieve_node(state: VeriRAGState) -> VeriRAGState:
    """Hybrid-retrieve for the query (dense + sparse -> RRF -> rerank)."""
    corpus_only = bool(state.get("corpus_only"))
    resolved = state.get("resolved_doc_ids") or []
    # Scope to a single named document only when exactly one was resolved;
    # zero or several means search the whole corpus, so cross-document
    # comparison questions still see everything.
    doc_id_filter = resolved[0] if len(resolved) == 1 else None
    search_query = state.get("search_query") or state["query"]

    hits = hybrid_retrieve(
        search_query,
        corpus_only=corpus_only,
        doc_id_filter=doc_id_filter,
        session_id=state.get("session_id"),
    )
    local_count = len(hits)

    # Merge in any external material fetched for THIS query. Retrieval itself
    # never returns persisted external chunks, so this is the only route by
    # which a fresh fallback reaches generation -- and it cannot leak into a
    # later query, because the state is per-run.
    external_hits = state.get("external_hits") or []
    if external_hits and not corpus_only:
        hits = sorted(
            hits + external_hits,
            key=lambda hit: hit.rerank_score or 0.0,
            reverse=True,
        )[: settings.retrieval_final_top_k]

    top = hits[0].rerank_score if hits else None
    return {
        "hits": hits,
        "pipeline_log": [retrieval_event(hits)],
        "trace": [
            f"retrieve: {len(hits)} hits ({local_count} local"
            + (f" + {len(external_hits)} freshly fetched" if external_hits else "")
            + ")"
            + (f", top rerank_score={top:.6f}" if top is not None else "")
            + (" (corpus-only)" if corpus_only else "")
            + (f" (scoped to doc_id={doc_id_filter})" if doc_id_filter else "")
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

    return {
        "gate1_verdict": verdict,
        "llm_call_count": llm_calls,
        "trace": trace,
        "pipeline_log": [
            gate1_event(
                score=verdict.max_rerank_score,
                verdict=verdict.verdict,
                decision_path=verdict.decision_path,
                threshold_high=settings.gate1_high_threshold,
                threshold_low=settings.gate1_low_threshold,
                llm_reason=verdict.llm_reason,
            )
        ],
    }


@_timed("fetch_external")
def fetch_external_node(state: VeriRAGState) -> VeriRAGState:
    """Fetch external material once and make it searchable."""
    verdict = state.get("gate1_verdict")
    pre_fetch_max_score = verdict.max_rerank_score if verdict else 0.0

    # Search with the filename-stripped query: sending the raw text is what
    # fetched pages about summarizing emails for "summarize this paper
    # please: <file>.pdf" and polluted the store.
    search_query = state.get("search_query") or state["query"]
    documents = fetch_external(search_query)
    trace = [f"fetch_external: returned {len(documents)} document(s)"]

    external_hits: list[SearchHit] = []
    if documents:
        # Still ingested, which chunks/contextualizes/embeds them and doubles
        # as the URL-keyed dedup cache. Retrieval will never surface them
        # again, so they are also pulled back by doc_id right here and scored
        # for this query only.
        results = ingest_documents(documents)
        ingested_chunks = sum(
            r.num_text_chunks + r.num_figure_chunks for r in results
        )
        trace.append(
            f"fetch_external: ingested {len(results)} source(s), "
            f"{ingested_chunks} chunk(s)"
        )

        candidates: list[SearchHit] = []
        for result in results:
            candidates.extend(get_chunks_by_doc_id(result.doc_id))
        external_hits = score_external_hits(search_query, candidates)
        if external_hits:
            best = external_hits[0].rerank_score or 0.0
            trace.append(
                f"fetch_external: scored {len(external_hits)} external chunk(s) "
                f"in-memory for this query (best rerank_score={best:.6f})"
            )

    return {
        "external_fetch_attempted": True,
        "pre_fetch_max_score": pre_fetch_max_score,
        "external_hits": external_hits,
        "trace": trace,
    }


@_timed("generate")
def generate_node(state: VeriRAGState) -> VeriRAGState:
    """Generate a cited answer from the retrieved hits."""
    hits = state.get("hits", [])
    answer = generate_answer(state["query"], hits, state["answer_format"])
    # Re-derive the exact context block generate_answer sent to the model.
    # build_context_block is a pure function of the hits, so this is the same
    # string; logged for the Pipeline Log tab only.
    context_block, _ = build_context_block(hits)
    return {
        "generated_answer": answer,
        "llm_call_count": state.get("llm_call_count", 0) + 1,
        "pipeline_log": [context_event(hits, context_block)],
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
    hit_types = {
        hit.chunk_id: hit.metadata.get("content_type") for hit in answer.hits_used
    }
    return {
        "gate2_verdict": verdict,
        "pipeline_log": [gate2_event(verdict.claims, hit_types)],
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
    # A false_hedge needs the opposite steer from an unsupported assertion:
    # answer the denied topic rather than omit a claim. Split the two so each
    # gets its own instruction.
    avoid = [
        claim.claim_text
        for claim in verdict.unsupported_claims
        if claim.verdict != "false_hedge"
    ]
    false_hedge_topics = [
        claim.probe_topic
        for claim in verdict.unsupported_claims
        if claim.verdict == "false_hedge" and claim.probe_topic
    ]

    answer = generate_answer(
        state["query"],
        state.get("hits", []),
        state["answer_format"],
        avoid_claims=avoid,
        false_hedge_topics=false_hedge_topics,
    )
    attempt = state.get("regeneration_count", 0) + 1
    reason = f"Gate 2 found {len(avoid)} unsupported claim(s)"
    if false_hedge_topics:
        reason += f" and {len(false_hedge_topics)} false hedge(s)"
    return {
        "generated_answer": answer,
        "regeneration_count": attempt,
        "pipeline_log": [
            regen_event(attempt, settings.max_critic_iterations, reason)
        ],
        "llm_call_count": state.get("llm_call_count", 0) + 1,
        "trace": [
            f"regenerate (attempt {attempt}/{settings.max_critic_iterations}): "
            f"avoiding {len(avoid)} unverified claim(s), "
            f"re-addressing {len(false_hedge_topics)} false hedge(s)"
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
        "pipeline_log": [answer_event(answer.answer_text)],
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
        "pipeline_log": [abstain_event(reason)],
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
        "pipeline_log": [abstain_event(reason)],
        "trace": [
            "abstain_ungrounded: refused to return an unverified answer "
            f"({len(unsupported)} unsupported claim(s))"
        ],
    }
