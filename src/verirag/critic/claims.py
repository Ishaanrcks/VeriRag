"""Decompose a generated answer into atomic, independently-checkable claims.

Gate 2 cannot usefully run NLI over a whole multi-paragraph answer, so one
DeepSeek call splits it into atomic claims, records which ``[n]`` citation(s)
each relies on, and classifies each claim:

* ``factual``  -- a concrete statement about document content; verified by NLI.
* ``absence``  -- asserts a source does NOT mention something. NLI entailment
  cannot confirm a negative from a passage that simply never discusses the
  topic, so Gate 2 handles these separately.
* ``meta``     -- a statement about the *provided passages* rather than a fact
  drawn from them (commentary on retrieval completeness); excluded from
  groundedness scoring entirely.

Parsing is defensive: any failure degrades to a single whole-answer factual
claim rather than raising.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from verirag.generation.generator import GeneratedAnswer
from verirag.llm.client import chat_completion

logger = logging.getLogger(__name__)

ClaimType = Literal["factual", "absence", "meta"]

_VALID_CLAIM_TYPES: set[str] = {"factual", "absence", "meta"}
_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Claim:
    """One atomic claim lifted out of a generated answer."""

    text: str
    claim_type: ClaimType
    cited_numbers: list[int] = field(default_factory=list)


_EXTRACTION_PROMPT = """\
Decompose the answer below into atomic, independently checkable claims.

For each claim produce an object with these keys:
- "text": the claim as a single self-contained sentence. Resolve pronouns so it
  stands alone. Do NOT include the [n] citation markers inside this text.
- "claim_type": exactly one of:
    - "factual": a concrete statement about document content that could be
      checked against a source passage.
    - "absence": asserts that a specific source does NOT mention or contain
      something. Example: "The DOCX fixture does not mention any revenue chart."
    - "meta": a statement about the provided passages or context themselves --
      what they do not describe, or the limits of the available context --
      rather than a fact drawn from them. Example: "The passages do not
      describe how the gates are implemented."
- "cited_numbers": a JSON array of integers, the citation numbers taken from
  the [n] markers attached to that claim in the answer. Use [] if it has none.

Answer to decompose:
<answer>
{answer_text}
</answer>

Respond with ONLY a JSON array of those objects. No markdown fences, no \
commentary, no trailing text."""


def _citation_numbers_in(text: str) -> list[int]:
    return sorted({int(number) for number in _CITATION_RE.findall(text)})


def _fallback_claims(answer: GeneratedAnswer, reason: str) -> list[Claim]:
    """Degrade to one whole-answer factual claim; never raise."""
    logger.warning(
        "Claim extraction falling back to a single whole-answer claim (%s).",
        reason,
    )
    return [
        Claim(
            text=answer.answer_text,
            claim_type="factual",
            cited_numbers=_citation_numbers_in(answer.answer_text),
        )
    ]


def _isolate_json_array(response: str) -> str | None:
    """Pull the JSON array out of a response, tolerating markdown fences."""
    text = response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _coerce_claim(item: object) -> Claim | None:
    """Validate/coerce one parsed JSON object into a Claim, or drop it."""
    if not isinstance(item, dict):
        return None

    text = str(item.get("text", "")).strip()
    if not text:
        return None

    raw_type = str(item.get("claim_type", "")).strip().lower()
    if raw_type not in _VALID_CLAIM_TYPES:
        logger.warning(
            "Unrecognized claim_type %r; treating claim as factual.", raw_type
        )
        raw_type = "factual"

    cited: list[int] = []
    raw_cited = item.get("cited_numbers", [])
    if isinstance(raw_cited, (list, tuple)):
        for value in raw_cited:
            try:
                cited.append(int(value))
            except (TypeError, ValueError):
                continue

    return Claim(
        text=text,
        claim_type=raw_type,  # type: ignore[arg-type]
        cited_numbers=sorted(set(cited)),
    )


def extract_claims(answer: GeneratedAnswer) -> list[Claim]:
    """Split ``answer`` into atomic classified claims (one LLM call)."""
    if not answer.answer_text.strip():
        return []

    response = chat_completion(
        messages=[
            {
                "role": "user",
                "content": _EXTRACTION_PROMPT.format(
                    answer_text=answer.answer_text
                ),
            }
        ],
        temperature=0.0,
        max_tokens=2000,
    )
    if response is None:
        return _fallback_claims(answer, "model call failed")

    payload = _isolate_json_array(response)
    if payload is None:
        return _fallback_claims(answer, "no JSON array found in response")

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        return _fallback_claims(answer, f"JSON decode error: {exc}")

    if not isinstance(parsed, list):
        return _fallback_claims(answer, "parsed JSON was not a list")

    claims = [claim for claim in (_coerce_claim(item) for item in parsed) if claim]
    if not claims:
        return _fallback_claims(answer, "no valid claims after coercion")
    return claims
