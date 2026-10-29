"""Decompose a generated answer into atomic, independently-checkable claims.

Gate 2 cannot usefully run NLI over a whole multi-paragraph answer, so one
DeepSeek call splits it into atomic claims, records which ``[n]`` citation(s)
each relies on, and classifies each claim:

* ``factual``  -- a concrete statement about document content; verified by NLI.
* ``absence``  -- asserts a source does NOT mention something. NLI entailment
  cannot confirm a negative from a passage that simply never discusses the
  topic, so Gate 2 handles these separately.
* ``meta``     -- a statement about the *provided passages* rather than a fact
  drawn from them (commentary on retrieval completeness, or a characterisation
  of a passage such as "Passage 2 is an unrelated course document"); excluded
  from groundedness scoring entirely.
* ``structural`` -- a statement about the document's ORGANISATION: that a
  heading, section or chapter with a given title exists, or what such a section
  covers. Entailment is the wrong test (it asks whether a sentence pair is
  semantically consistent, not whether a title literally appears), so Gate 2
  verifies these by presence in the cited text instead.

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

ClaimType = Literal["factual", "absence", "meta", "unscorable", "structural"]

_VALID_CLAIM_TYPES: set[str] = {
    "factual", "absence", "meta", "unscorable", "structural",
}
_CITATION_RE = re.compile(r"\[(\d+)\]")

# Absence/hedge detection. Regex-based on purpose: it catches hedges REGARDLESS
# of how the LLM classifier labelled them. That matters because the generator
# is now instructed to cite every positive sentence, which pushes an uncited
# hedge ("the documents do not discuss X") toward the "factual" bucket -- so we
# cannot rely on the model tagging it "absence". A matched hedge is forced to
# claim_type="absence" and its denied TOPIC is extracted for Gate 2 to probe.
_HEDGE_RE = re.compile(
    r"\b("
    r"do(?:es)?\s+not\s+(?:explicitly\s+|specifically\s+|directly\s+)?"
    r"(?:state|mention|discuss|address|cover|describe|contain|provide|"
    r"specify|say|include|report|define|list)"
    r"|do(?:es)?n't\s+(?:state|mention|discuss|address|cover|describe|"
    r"contain|provide|specify|say|include)"
    r"|is\s+not\s+(?:discussed|mentioned|addressed|covered|stated|described|"
    r"specified|provided|included|present|found|defined)"
    r"|are\s+not\s+(?:discussed|mentioned|addressed|covered|stated|described|"
    r"specified|provided|included|present|found|defined)"
    r"|was\s+not\s+(?:discussed|mentioned|addressed|covered|found)"
    r"|were\s+not\s+(?:discussed|mentioned|addressed|covered|found)"
    r"|fails?\s+to\s+(?:mention|discuss|address|state|describe|specify)"
    r"|lacks?\s+(?:any\s+)?(?:information|mention|discussion|detail|details|"
    r"reference)\s+(?:of|about|on|regarding)"
    r"|no\s+(?:information|mention|discussion|detail|details|reference|"
    r"data)\s+(?:about|on|regarding|of|concerning)"
    r"|there\s+(?:is|are|was|were)\s+no\s+(?:information|mention|discussion|"
    r"reference|detail|details)"
    r"|not\s+(?:covered|mentioned|discussed|addressed|stated|found|present|"
    r"available|provided|included|specified)"
    r")\b",
    re.IGNORECASE,
)
# Leading filler words stripped off the extracted topic, run until none remain.
_TOPIC_LEADER_RE = re.compile(
    r"^(?:about|on|of|regarding|concerning|that|whether|how|any|the|this|"
    r"these|those|it|them|such|information|anything|details?)\b\s*",
    re.IGNORECASE,
)


def _absence_topic(text: str) -> str | None:
    """Denied topic if ``text`` is an absence/hedge claim, else ``None``.

    Returns ``None`` when the text is not a hedge at all. Returns the topic
    (possibly an empty string, when the hedge names no explicit subject) when
    it is. Callers treat "not None" as "this is an absence claim".
    """
    match = _HEDGE_RE.search(text)
    if match is None:
        return None
    tail = text[match.end():].strip(" .,:;\"'()[]")
    previous = None
    while tail and tail != previous:
        previous = tail
        tail = _TOPIC_LEADER_RE.sub("", tail).strip(" .,:;\"'()[]")
    return tail


# Structural-claim detection. Like the hedge regex above this is deterministic on
# purpose: the LLM labels these inconsistently (the same "section titled X"
# sentence came back "factual" in one run and "meta" in another), and a claim
# mislabelled "factual" is scored by NLI, which cannot say whether a heading
# literally appears in a passage (measured: 0.0035-0.22 on headings that were
# there). A claim is structural when it QUOTES a title and talks about a
# heading/section/chapter.
_QUOTED_RE = re.compile(r"[\"\u201c]([^\"\u201c\u201d]{6,}?)[\"\u201d]")
_STRUCTURAL_WORD_RE = re.compile(
    r"\b(?:sections?|chapters?|headings?|headed|titled|titles?|headers?|subsections?)\b",
    re.IGNORECASE,
)
# A passage being characterised, not a fact drawn from it: "Passage 2 is an
# unrelated business management course document."
_PASSAGE_IDENTITY_RE = re.compile(
    r"^\s*(?:the\s+)?(?:(?:first|second|third|fourth|fifth)\s+)?passages?(?:\s+\d+)?"
    r"\s+(?:is|are|appears?\s+to\s+be|seems?\s+to\s+be)\b",
    re.IGNORECASE,
)


def structural_titles(text: str) -> list[str] | None:
    """Quoted titles if ``text`` is a structural claim, else ``None``.

    Structural = it quotes a title AND talks about a heading/section/chapter.
    A number outside the quotes means the claim asserts a measurement ("section
    "Results" shows 92%"), which presence cannot verify, so it is left to NLI.
    """
    quotes = [q.strip() for q in _QUOTED_RE.findall(text)]
    if not quotes or not _STRUCTURAL_WORD_RE.search(text):
        return None
    outside = _QUOTED_RE.sub(" ", text)
    outside = re.sub(r"\bpassages?\s+\d+\b|\[\d+\]", " ", outside, flags=re.IGNORECASE)
    if re.search(r"\d", outside):
        return None
    return quotes


@dataclass
class Claim:
    """One atomic claim lifted out of a generated answer."""

    text: str
    claim_type: ClaimType
    cited_numbers: list[int] = field(default_factory=list)
    # For absence claims: the denied subject (e.g. "the six categories of the
    # cognitive process dimension"), which Gate 2 probes for in the evidence.
    # Empty string = detected as a hedge but no explicit topic to probe.
    topic: str | None = None
    # For structural claims: the quoted heading/section titles to find verbatim.
    titles: list[str] = field(default_factory=list)


_EXTRACTION_PROMPT = """\
Decompose the answer below into atomic, independently checkable claims.

For each claim produce an object with these keys:
- "text": the claim as a single self-contained sentence. Resolve pronouns so it
  stands alone. Do NOT include the [n] citation markers inside this text.
  Never refer to passages by number ("Passage 1 ...") -- those numbers exist
  only in the prompt and mean nothing to the verifier. A claim that merely
  characterises a passage ("Passage 2 is unrelated") is "meta".
  State the claim as a BARE PROPOSITION. Strip any attributive wrapper: write
  "ExpressGNN combines Markov Logic Networks and graph neural networks", NOT
  "The paper states that ExpressGNN combines...". The wrapper turns the claim
  into a statement about the text rather than about the subject, which the
  verifier cannot check.
- "claim_type": exactly one of:
    - "factual": a concrete statement about document content that could be
      checked against a source passage.
    - "absence": asserts that a specific source does NOT mention or contain
      something. Example: "The DOCX fixture does not mention any revenue chart."
    - "meta": a statement about the provided passages or context themselves --
      what they do not describe, or the limits of the available context --
      rather than a fact drawn from them. Example: "The passages do not
      describe how the gates are implemented."
    - "structural": a statement about how the document is ORGANISED -- that a
      heading, section or chapter with a given title exists or is mentioned, or
      what such a section covers. Example: 'The section "Methods" covers data
      collection.' Quote the title exactly as it appears.
    - "unscorable": content that is not a checkable proposition at all -- a
      formula or equation ("H = GNN(X, E)"), or a pointer to a figure, table
      or section rather than a statement of fact ("Figure 3 reports the token
      budget"). Use this whenever the claim's substance is an artifact in the
      document rather than an assertion about the world.
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
    topic = _absence_topic(answer.answer_text)
    return [
        Claim(
            text=answer.answer_text,
            claim_type="absence" if topic is not None else "factual",
            cited_numbers=_citation_numbers_in(answer.answer_text),
            topic=topic,
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

    # Regex hedge detection overrides the LLM label: a matched hedge is an
    # absence claim whatever the model called it, so it reaches Gate 2's
    # contradiction probe instead of being scored as a positive assertion.
    topic = _absence_topic(text)
    if topic is not None:
        raw_type = "absence"

    # Only a claim the model called "factual" (or "structural") is re-routed:
    # "meta" and "absence" are already excluded from entailment scoring, and an
    # exclusivity claim ("the ONLY heading ... is X") is commentary on the
    # retrieved context that a presence check cannot verify.
    titles: list[str] = []
    if raw_type in ("factual", "structural"):
        found = structural_titles(text)
        if found is not None:
            raw_type, titles = "structural", found
        elif raw_type == "factual" and _PASSAGE_IDENTITY_RE.match(text):
            raw_type = "meta"

    return Claim(
        text=text,
        claim_type=raw_type,  # type: ignore[arg-type]
        cited_numbers=sorted(set(cited)),
        topic=topic,
        titles=titles,
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
