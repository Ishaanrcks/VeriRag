"""Gate 1: coverage check, run BEFORE generation.

Gate 1 asks a single question: do the retrieved passages contain enough
information to answer the query at all? Phase 4 established that the
cross-encoder's ``rerank_score`` -- not the rank-based ``fused_score`` -- is
the usable *absolute* relevance signal, so that is what this gate reads.

Three decision paths, cheapest first:
  * ``threshold_high``  -- clearly relevant evidence; no LLM call.
  * ``threshold_low``   -- clearly nothing relevant; no LLM call.
  * ``llm_escalation``  -- borderline band only; one LLM judge call.

This gate only *emits* a verdict. Acting on it (fetching external sources,
abstaining) is Phase 8's branching logic, deliberately not done here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from langsmith import traceable

from verirag.config import settings
from verirag.llm.client import chat_completion
from verirag.retrieval.types import SearchHit

logger = logging.getLogger(__name__)

Verdict = Literal["sufficient", "insufficient"]
DecisionPath = Literal["threshold_high", "threshold_low", "llm_escalation"]


@dataclass
class Gate1Verdict:
    """Outcome of the coverage check, plus why it was reached."""

    verdict: Verdict
    decision_path: DecisionPath
    max_rerank_score: float
    # max - min rerank_score across the hits. Recorded for Phase 12
    # calibration only; it does NOT influence the decision in this MVP.
    score_spread: float
    # Set only when decision_path == "llm_escalation".
    llm_reason: str | None = None


_JUDGE_PROMPT = """\
You are judging whether retrieved passages contain enough information to \
answer a question.

Question: {query}

Retrieved passages:
{passages}

Decide whether these passages, taken together, contain enough information to \
answer the question. Apply these tests:

1. ON-TOPIC: the passages must be about the question's actual subject, not \
merely about something in the same general category. Two things can both be \
called a "taxonomy" (or a "model", a "dataset", a "framework") and still be \
completely different things -- a passage about Bloom's educational taxonomy \
does NOT help answer a question about a taxonomy of graph neural networks. \
Shared vocabulary is not shared subject matter.
2. PRESUPPOSITION: if the question assumes something the passages do not \
establish, that is INSUFFICIENT -- say which assumption fails.
3. PARTIAL ANSWERS COUNT: answer SUFFICIENT if the passages support ANY \
correct answer, even an incomplete one. If the passages state the thing being \
asked about -- a definition, a count, a name, a method -- that is SUFFICIENT \
even if they do not elaborate on it further.

Answer INSUFFICIENT only when the passages are about a DIFFERENT subject than \
the question, or contain nothing from which any correct partial answer could \
be constructed. Do NOT answer INSUFFICIENT merely because the passages omit \
some detail a fuller answer might include, or because you personally know \
more about the topic than they state -- an incomplete but correct answer is \
still an answer, and this gate only decides whether answering is possible at \
all.

Respond in EXACTLY this format and nothing else:
Line 1: the single word SUFFICIENT or INSUFFICIENT
Line 2: a one-sentence reason

Do not add any other text, markdown, headings, or formatting."""


def _score_of(hit: SearchHit) -> float:
    """rerank_score, treating an unreranked hit as zero relevance."""
    return hit.rerank_score if hit.rerank_score is not None else 0.0


def _llm_judge_coverage(query: str, hits: list[SearchHit]) -> tuple[Verdict, str]:
    """Ask the chat model to judge coverage; parse strictly, fail to abstain.

    Any ambiguity -- an unreachable model, or a response whose first line is
    not exactly SUFFICIENT/INSUFFICIENT -- resolves to "insufficient". Failing
    toward abstention is the safe direction; a false "sufficient" would let an
    ungrounded answer through.
    """
    passages = "\n\n".join(
        f"[{i}] (rerank_score={_score_of(hit):.4f})\n{hit.raw_text}"
        for i, hit in enumerate(hits, start=1)
    )

    response = chat_completion(
        messages=[
            {
                "role": "user",
                "content": _JUDGE_PROMPT.format(query=query, passages=passages),
            }
        ],
        temperature=0.0,
        max_tokens=150,
    )

    if response is None:
        logger.warning(
            "Gate 1 escalation: no response from judge model; "
            "defaulting to insufficient."
        )
        return (
            "insufficient",
            "Judge model call failed; defaulted to insufficient (fail-safe).",
        )

    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    # Normalize away punctuation/markdown so "**SUFFICIENT**" still parses.
    # Exact equality matters here: "INSUFFICIENT" *contains* "SUFFICIENT", so a
    # substring test would silently invert the verdict.
    first = re.sub(r"[^A-Za-z]", "", lines[0]).upper() if lines else ""
    reason = lines[1] if len(lines) > 1 else "(no reason provided)"

    if first == "SUFFICIENT":
        return "sufficient", reason
    if first == "INSUFFICIENT":
        return "insufficient", reason

    logger.warning(
        "Gate 1 escalation: unparseable judge verdict %r; "
        "defaulting to insufficient.",
        response[:200],
    )
    snippet = lines[0][:120] if lines else "(empty response)"
    return (
        "insufficient",
        f"Unparseable judge response, defaulted to insufficient "
        f"(fail-safe). First line was: {snippet!r}",
    )


@traceable(name="check_coverage")
def check_coverage(query: str, hits: list[SearchHit]) -> Gate1Verdict:
    """Judge whether ``hits`` cover ``query`` well enough to answer it."""
    if not hits:
        logger.info("Gate 1: no hits retrieved -> insufficient (no LLM call).")
        return Gate1Verdict(
            verdict="insufficient",
            decision_path="threshold_low",
            max_rerank_score=0.0,
            score_spread=0.0,
            llm_reason=None,
        )

    scores = [_score_of(hit) for hit in hits]
    max_score = max(scores)
    score_spread = max_score - min(scores)

    # BENCHMARK FLAG gate1_advisory: Gate 1 stops being able to decline on the
    # strength of the LLM judge. Measured on tier-2, the judge declined 9 questions
    # in the escalation band and was wrong on 8 -- the annotated evidence page had
    # been retrieved for 7 of those 8. Advisory mode keeps the hard floor (nothing
    # relevant retrieved at all) and records the score as always, but lets anything
    # above the floor through to generation, where Gate 2 still has to verify it.
    if settings.gate1_advisory and max_score > settings.gate1_low_threshold:
        logger.info("Gate 1 (advisory): max_rerank_score=%.4f > %.4f -> sufficient, "
                    "no LLM judge.", max_score, settings.gate1_low_threshold)
        return Gate1Verdict(
            verdict="sufficient",
            decision_path="threshold_high",
            max_rerank_score=max_score,
            score_spread=score_spread,
            llm_reason=None,
        )

    if max_score >= settings.gate1_high_threshold:
        logger.info(
            "Gate 1: max_rerank_score=%.4f >= %.4f -> sufficient (no LLM call).",
            max_score, settings.gate1_high_threshold,
        )
        return Gate1Verdict(
            verdict="sufficient",
            decision_path="threshold_high",
            max_rerank_score=max_score,
            score_spread=score_spread,
            llm_reason=None,
        )

    if max_score <= settings.gate1_low_threshold:
        logger.info(
            "Gate 1: max_rerank_score=%.4f <= %.4f -> insufficient "
            "(no LLM call).",
            max_score, settings.gate1_low_threshold,
        )
        return Gate1Verdict(
            verdict="insufficient",
            decision_path="threshold_low",
            max_rerank_score=max_score,
            score_spread=score_spread,
            llm_reason=None,
        )

    logger.info(
        "Gate 1: max_rerank_score=%.4f in borderline band (%.4f, %.4f) "
        "-> escalating to LLM judge.",
        max_score, settings.gate1_low_threshold, settings.gate1_high_threshold,
    )
    verdict, reason = _llm_judge_coverage(query, hits)
    return Gate1Verdict(
        verdict=verdict,
        decision_path="llm_escalation",
        max_rerank_score=max_score,
        score_spread=score_spread,
        llm_reason=reason,
    )
