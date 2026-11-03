"""An UNVERIFIED candidate answer, offered when the gated pipeline declines.

Measured on the stored C2 and R2 runs: of the 78 questions where the gated
pipeline returns no verified answer, the ungated pipeline produces an answer for
40, and 20 of those are correct -- 50% precision. Half of them are wrong, and four
of the wrong ones are questions whose correct answer is "not answerable", so this
can never be presented as an answer. It is a lead for a reader who wants one, and
it is kept strictly apart from the verified answer.

The candidate NEVER changes the verified answer, never enters the Verified facts,
and is never produced when Gate 1 declined for want of any relevant evidence --
there is nothing to offer in that case, and offering something would be inventing it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

UNVERIFIED_LABEL = "UNVERIFIED"

REASON_DERIVED = "derived by the model (value not in the sources)"
REASON_CAPTION = "read from a figure caption only"
REASON_UNSUPPORTED = "no supporting passage found for a claim"
REASON_OTHER = "other"

_CAPTION_TYPES = ("figure_caption", "image_caption")
_NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


@dataclass
class CandidateAnswer:
    """What the pipeline would have said, had nothing been verifying it."""

    answer: str
    reason: str
    sources: list[dict] = field(default_factory=list)
    label: str = UNVERIFIED_LABEL

    def as_dict(self) -> dict:
        return dict(label=self.label, answer=self.answer, reason=self.reason,
                    sources=list(self.sources))


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in _NUM_RE.finditer(text or "")}


def classify_reason(answer_text: str, gate2_verdict, hits: list) -> str:
    """Why this answer was not verified, in words a reader can act on.

    Checked in order of how much it should worry the reader: a value the model
    worked out itself is the least trustworthy, so it is reported first even when
    Gate 2 also had other complaints.
    """
    claims = list(getattr(gate2_verdict, "claims", None) or [])
    verdicts = {getattr(c, "verdict", "") for c in claims}

    # A number in the answer that appears in NO retrieved passage was computed,
    # not read -- the single most common wrong-candidate shape in the measurement.
    retrieved = set()
    for hit in hits:
        retrieved |= _numbers(getattr(hit, "raw_text", "") or "")
    invented = _numbers(answer_text) - retrieved
    if invented:
        return REASON_DERIVED
    if "caption_only_uncertified" in verdicts:
        return REASON_CAPTION
    if {"not_supported", "external_unsupported"} & verdicts:
        return REASON_UNSUPPORTED
    return REASON_OTHER


def describe_sources(hits: list, limit: int = 6) -> list[dict]:
    """The chunks the candidate drew on, with their pages, de-duplicated."""
    from pathlib import Path as _Path

    out, seen = [], set()
    for hit in hits:
        meta = getattr(hit, "metadata", None) or {}
        key = (str(meta.get("source_path")), meta.get("page"), str(meta.get("content_type")))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(document=_Path(str(meta.get("source_path") or "")).name,
                        page=meta.get("page"), type=str(meta.get("content_type"))))
        if len(out) >= limit:
            break
    return out


def derivation_note(answer_text: str, hits: list) -> str:
    """"derived from X and Y" with the inputs and their pages, for a computed value.

    A derived figure is shown as what it was worked out FROM, so the reader sees the
    inputs and can check them, rather than being handed a number as if it were a fact.
    """
    from pathlib import Path as _Path

    answer_numbers = _numbers(answer_text)
    retrieved: set[str] = set()
    for hit in hits:
        retrieved |= _numbers(getattr(hit, "raw_text", "") or "")
    # Only a value the passages do NOT state was derived. Without this the note
    # would claim a derivation for any answer that merely quotes a number.
    if not (answer_numbers - retrieved):
        return ""
    found: list[str] = []
    for hit in hits:
        meta = getattr(hit, "metadata", None) or {}
        shared = _numbers(getattr(hit, "raw_text", "") or "") & answer_numbers
        for value in sorted(shared):
            entry = f"{value} (p{meta.get('page')} of {_Path(str(meta.get('source_path') or '')).name})"
            if entry not in found:
                found.append(entry)
        if len(found) >= 4:
            break
    if not found:
        return ""
    return "derived from " + " and ".join(found[:4])


def build_candidate(answer_text: str, gate2_verdict, hits: list) -> CandidateAnswer | None:
    """A labelled candidate, or None when there is nothing honest to offer."""
    text = (answer_text or "").strip()
    if not text:
        return None
    reason = classify_reason(text, gate2_verdict, hits)
    if reason == REASON_DERIVED:
        note = derivation_note(text, hits)
        if note:
            text = f"{text}\n\n[{note}]"
    logger.info("unverified candidate offered (%s)", reason)
    return CandidateAnswer(answer=text, reason=reason, sources=describe_sources(hits))
