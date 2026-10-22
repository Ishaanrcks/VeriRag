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

from langsmith import traceable

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


def _json_array_region(response: str) -> str | None:
    """Return everything from the opening ``[`` onward, tolerating fences.

    Deliberately does NOT trim to the last ``]``: in a truncated response the
    last ``]`` often belongs to a nested ``cited_numbers`` array rather than
    closing the top-level one, which is what produced the malformed-JSON
    failures this function's callers now recover from.
    """
    text = response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("[")
    if start == -1:
        return None
    return text[start:]


def _recover_complete_objects(array_region: str) -> list[str]:
    """Extract the complete top-level ``{...}`` objects from an array body.

    Walks the text tracking brace depth while respecting string literals and
    escapes, so a truncated response still yields every claim that arrived
    intact before the cut-off. Anything after the truncation point is simply
    not emitted.
    """
    objects: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False

    for index, char in enumerate(array_region):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(array_region[start : index + 1])
                    start = None

    return objects


def _claims_from_items(items: list) -> list[Claim]:
    return [claim for claim in (_coerce_claim(item) for item in items) if claim]


def _extraction_max_tokens(answer_text: str) -> int:
    """Scale the claim-extraction budget to the length of the input.

    The JSON decomposition restates essentially all of the answer plus
    per-claim scaffolding, so in practice the output runs roughly two to three
    times the input length in characters. A fixed budget that suits a concise
    QA answer therefore truncates a whole-document summary mid-array. The
    multiplier below leaves real headroom (a generous ceiling costs nothing
    unless the model actually uses it); the floor keeps short answers cheap
    and the cap stays within the model's output limit.
    """
    return min(8000, max(2000, int(len(answer_text) * 0.75)))


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


@traceable(name="extract_claims")
def extract_claims(answer: GeneratedAnswer) -> list[Claim]:
    """Split ``answer`` into atomic classified claims (one LLM call)."""
    if not answer.answer_text.strip():
        return []

    max_tokens = _extraction_max_tokens(answer.answer_text)
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
        max_tokens=max_tokens,
    )
    if response is None:
        return _fallback_claims(answer, "model call failed")

    region = _json_array_region(response)
    if region is None:
        return _fallback_claims(answer, "no JSON array found in response")

    # 1. Happy path: the array closed cleanly and parses whole.
    closing = region.rfind("]")
    if closing > 0:
        try:
            parsed = json.loads(region[: closing + 1])
            if isinstance(parsed, list):
                claims = _claims_from_items(parsed)
                if claims:
                    return claims
        except json.JSONDecodeError:
            pass  # fall through to truncation recovery

    # 2. Truncation recovery: salvage every claim that arrived intact rather
    #    than collapsing the whole answer into one unverifiable mega-claim.
    #    A single claim spanning a long answer can never be entailed by one
    #    chunk, so the old fallback guaranteed a Gate 2 failure even for a
    #    perfectly grounded summary.
    recovered: list[Claim] = []
    for object_text in _recover_complete_objects(region):
        try:
            item = json.loads(object_text)
        except json.JSONDecodeError:
            continue
        claim = _coerce_claim(item)
        if claim:
            recovered.append(claim)

    if recovered:
        logger.warning(
            "Claim-extraction response was truncated or malformed; recovered "
            "%d complete claim(s) from %d chars of output (budget was %d "
            "tokens). Claims after the cut-off were lost; raise the budget if "
            "this recurs.",
            len(recovered), len(response), max_tokens,
        )
        return recovered

    return _fallback_claims(answer, "no recoverable claims in response")
