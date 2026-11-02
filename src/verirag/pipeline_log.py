"""Structured per-query pipeline trace, for the UI's "Pipeline Log" tab.

Each graph node appends ``LogEvent`` objects alongside its normal work (see
``graph/state.py`` -- ``pipeline_log`` accumulates like ``trace``). Nothing here
changes pipeline behavior; it only records what happened in a form the UI can
render as compact tables rather than raw text.

This module is deliberately pure (stdlib only, no model/config imports): the
builders take plain objects and primitives, so they are cheap to unit-test and
cannot create import cycles with the graph.

Storage vs display: events hold the data at useful fidelity (snippets ~200
chars, full claim text); the ``*_rows`` helpers truncate for display. The one
deliberately untruncated value is ``context`` detail ``full_context`` -- the
exact string handed to the LLM.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

Stage = Literal[
    "retrieval", "context", "gate1", "regen", "gate2", "answer", "abstain", "error"
]


@dataclass
class LogEvent:
    stage: Stage
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


def new_log() -> list[LogEvent]:
    return []


# --------------------------------------------------------------------------
# small formatting helpers
# --------------------------------------------------------------------------
SNIPPET_STORE_CHARS = 200  # stored per chunk
SNIPPET_DISPLAY_CHARS = 100  # shown in the retrieval table
CLAIM_DISPLAY_CHARS = 80  # shown in the Gate 2 table


def truncate(text: str | None, limit: int) -> str:
    """Collapse whitespace and cut to ``limit`` chars, ending in an ellipsis."""
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: max(0, limit - 1)].rstrip() + "…"


def doc_name(metadata: dict[str, Any] | None) -> str:
    """Short document label: a filename, or a host for web sources. Never a path."""
    metadata = metadata or {}
    source = str(metadata.get("source_path") or "")
    if metadata.get("source_type") == "external" or source.startswith(
        ("http://", "https://")
    ):
        parsed = urlparse(source)
        return parsed.netloc or source or "web"
    return os.path.basename(source) or "unknown"


def type_label(content_type: str | None) -> str:
    """Normalize a stored content_type to text | table | image_caption."""
    if content_type == "table":
        return "table"
    if content_type == "figure_caption":
        return "image_caption"
    return "text"


def type_display(type_value: str) -> str:
    """Badge-style label shown in tables ('image_caption' -> 'image caption')."""
    return type_value.replace("_", " ")


# --------------------------------------------------------------------------
# event builders
# --------------------------------------------------------------------------
def describe_hit(rank: int, hit: Any) -> dict[str, Any]:
    """Compact, JSON-friendly description of one retrieved chunk.

    ``score`` is the cross-encoder rerank score when present (it orders the list
    and is what Gate 1 reads), falling back to the RRF fused score; both raw
    values are kept for transparency.
    """
    metadata = getattr(hit, "metadata", None) or {}
    rerank = getattr(hit, "rerank_score", None)
    fused = getattr(hit, "fused_score", None)
    return {
        "rank": rank,
        "doc_name": doc_name(metadata),
        "page": metadata.get("page"),
        "type": type_label(metadata.get("content_type")),
        "score": rerank if rerank is not None else fused,
        "rerank_score": rerank,
        "rrf_score": fused,
        "snippet": truncate(getattr(hit, "raw_text", ""), SNIPPET_STORE_CHARS),
    }


def _chunk_list(hits: list[Any]) -> list[dict[str, Any]]:
    return [describe_hit(i, hit) for i, hit in enumerate(hits, start=1)]


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def retrieval_event(hits: list[Any], summary_prefix: str = "Retrieved") -> LogEvent:
    chunks = _chunk_list(hits)
    docs = {c["doc_name"] for c in chunks}
    return LogEvent(
        stage="retrieval",
        summary=f"{summary_prefix} {_plural(len(chunks), 'chunk')} "
                f"from {_plural(len(docs), 'document')}",
        detail={"chunks": chunks, "doc_count": len(docs), "chunk_count": len(chunks)},
    )


def context_event(
    hits: list[Any], full_context: str, mode: str = "retrieval"
) -> LogEvent:
    """The exact context string sent to the LLM, plus which chunks made it in.

    A hit counts as included when its text actually appears in the assembled
    context (parent text or an appended 'additional retrieved passage'), so this
    reflects the real string rather than assuming every retrieved chunk made it.
    """
    included_hits = [
        h for h in hits if (getattr(h, "raw_text", "") or "") in full_context
    ]
    included = _chunk_list(included_hits)
    return LogEvent(
        stage="context",
        summary=f"Context assembled: {len(full_context)} characters, "
                f"{len(included)} chunks included",
        detail={
            "full_context": full_context,
            "char_count": len(full_context),
            "chunks_included": included,
            "mode": mode,
        },
    )


def gate1_event(
    *,
    score: float,
    verdict: str,
    decision_path: str,
    threshold_high: float,
    threshold_low: float,
    llm_reason: str | None,
) -> LogEvent:
    """``verdict`` is Gate 1's own label ('sufficient' / 'insufficient')."""
    passed = verdict == "sufficient"
    # The number the score was compared against on the path that decided it.
    threshold = threshold_low if decision_path == "threshold_low" else threshold_high
    return LogEvent(
        stage="gate1",
        summary=f"Gate 1 {'pass' if passed else 'fail'} (score {score:.2f})",
        detail={
            "score": score,
            "threshold": threshold,
            "verdict": "pass" if passed else "fail",
            "decision_path": decision_path,
            "band": [threshold_low, threshold_high],
            "llm_reason": llm_reason,
        },
    )


def regen_event(iteration: int, max_iterations: int, reason: str) -> LogEvent:
    return LogEvent(
        stage="regen",
        summary=f"Regeneration attempt {iteration} of {max_iterations}",
        detail={"iteration": iteration, "max_iterations": max_iterations,
                "reason": reason},
    )


_GROUNDED = {"supported_single", "supported_combination", "supported_presence",
             "visually_verified", "supported_numeric", "visual_counted"}
_UNGROUNDED = {
    "not_supported", "external_unsupported", "caption_only_uncertified",
    "false_hedge",
}


def claim_verdict_bucket(label: str) -> str:
    """grounded | ungrounded | excluded (meta / unverifiable-hedge claims)."""
    if label in _GROUNDED:
        return "grounded"
    if label in _UNGROUNDED:
        return "ungrounded"
    return "excluded"


# Verdicts whose ``evidence_chunk_ids`` ARE the chunks that decided the verdict
# (the support set). Everything else either has no support (not_supported lists
# the chunks that were merely checked; meta/absence claims cite nothing), so a
# Source Type derived from those ids would describe chunks that did not support
# the claim.
_SUPPORT_SET_LABELS = {
    "supported_single", "supported_combination", "supported_presence",
    "visually_verified",
    "supported_numeric",
    "visual_counted",
    "caption_only_uncertified", "external_unsupported",
}


def gate2_event(claim_verdicts: list[Any], hit_types: dict[str, str]) -> LogEvent:
    """``hit_types`` maps chunk_id -> raw content_type of that chunk.

    ``source_type`` is built from the supporting chunks only; claims with no
    support set (not_supported, meta, absence) get None.
    """
    claims = []
    for cv in claim_verdicts:
        types: list[str] = []
        if cv.verdict in _SUPPORT_SET_LABELS:
            for chunk_id in getattr(cv, "evidence_chunk_ids", None) or []:
                label = type_label(hit_types.get(chunk_id))
                if label not in types:
                    types.append(label)
        claims.append({
            "text": cv.claim_text,
            "nli_score": cv.entailment_prob,
            "verdict": claim_verdict_bucket(cv.verdict),
            "label": cv.verdict,
            "source_type": " + ".join(types) if types else None,
        })
    ungrounded = sum(1 for c in claims if c["verdict"] == "ungrounded")
    return LogEvent(
        stage="gate2",
        summary=f"Gate 2 checked {len(claims)} claims, {ungrounded} ungrounded",
        detail={"claims": claims},
    )


def answer_event(text: str, partial: bool = False) -> LogEvent:
    return LogEvent(
        stage="answer",
        summary="Answered (partially verified)" if partial else "Answered",
        detail={"text": text, "partial": partial},
    )


def abstain_event(reason: str) -> LogEvent:
    return LogEvent(stage="abstain", summary="Abstained", detail={"reason": reason})


# --------------------------------------------------------------------------
# display rows (pure; the UI wraps these in a DataFrame)
# --------------------------------------------------------------------------
def _page_cell(page: Any) -> str:
    return "-" if page is None else str(page)


def retrieval_rows(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One compact row per chunk: Rank | Document | Page | Type | Score | Snippet."""
    return [
        {
            "Rank": c["rank"],
            "Document": c["doc_name"],
            "Page": _page_cell(c.get("page")),
            "Type": type_display(c["type"]),
            "Score": c.get("score"),
            "Snippet": truncate(c.get("snippet"), SNIPPET_DISPLAY_CHARS),
        }
        for c in chunks
    ]


def included_lines(chunks: list[dict[str, Any]]) -> list[str]:
    """One line per included chunk: document, page, type -- no text."""
    return [
        f"{c['doc_name']} - page {_page_cell(c.get('page'))} - "
        f"{type_display(c['type'])}"
        for c in chunks
    ]


_VERDICT_DISPLAY = {"grounded": "Grounded", "ungrounded": "Ungrounded",
                    "excluded": "Excluded"}


def gate2_rows(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Claim (80 chars) | Source Type | Score | Verdict.

    "Score" is the NLI entailment probability for NLI-decided verdicts and a
    token-overlap fraction for presence/absence verdicts, so it is not headed
    "NLI Score" (the UI captions this).
    """
    return [
        {
            "Claim": truncate(c["text"], CLAIM_DISPLAY_CHARS),
            "Source Type": (
                "-" if not c.get("source_type")
                else c["source_type"].replace("_", " ")
            ),
            "Score": c.get("nli_score"),
            "Verdict": _VERDICT_DISPLAY.get(c["verdict"], c["verdict"]),
        }
        for c in claims
    ]


def gate1_line(detail: dict[str, Any]) -> str:
    """One display line, e.g. 'PASS (score: 0.80 / threshold 0.50) - above ...'."""
    verdict = "PASS" if detail.get("verdict") == "pass" else "FAIL"
    head = (f"{verdict} (score: {detail['score']:.2f} / "
            f"threshold {detail['threshold']:.2f})")
    path = detail.get("decision_path")
    low, high = detail.get("band") or (None, None)
    if path == "threshold_high":
        return f"{head} - above the high threshold"
    if path == "threshold_low":
        return f"{head} - at or below the low threshold"
    reason = truncate(detail.get("llm_reason"), 160)
    band = f" in the borderline band {low:.2f}-{high:.2f}" if low is not None else ""
    return f"{head} - LLM judge decided{band}" + (f": {reason}" if reason else "")


def abstain_parts(reason: str, claim_limit: int = 160) -> tuple[str, list[str]]:
    """Split an abstention reason into a headline and truncated '- claim' bullets."""
    lines = [ln.strip() for ln in (reason or "").splitlines() if ln.strip()]
    if not lines:
        return "No reason recorded.", []
    headline = lines[0]
    bullets = [truncate(ln.lstrip("- ").strip(), claim_limit) for ln in lines[1:]]
    return headline, bullets
