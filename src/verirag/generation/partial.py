"""Partial-answer rendering: the answer, plus which facts in it were verified.

When Gate 2's final check leaves SOME claims verified and others not, the system
answers instead of abstaining. Nothing is stripped: the generator's text is
shown as written, followed by a "Verified facts" list of exactly the claims that
passed, each with the source it was checked against. A statement that is in the
answer but not in the list was not verified, and the closing line says so.

Pure and import-light on purpose (stdlib only): the eval harness and the
MMLongBench scorer import ``split_verified_facts`` to find this section in a
stored answer, so the heading lives in one place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VERIFIED_FACTS_HEADING = "**Verified facts**"
UNVERIFIED_NOTE = (
    "_Statements in the answer above that are not listed here could not be "
    "verified against the sources._"
)

# Gate 2 verdicts that mean "checked against the evidence and found supported":
# NLI entailment on one chunk or several, and lookup by presence (which covers
# structural claims, formula/figure references and the table-cell route).
VERIFIED_VERDICTS = frozenset(
    {"supported_single", "supported_combination", "supported_presence",
     "visually_verified", "supported_numeric", "visual_counted"}
)


@dataclass(frozen=True)
class VerifiedFact:
    """One claim that passed Gate 2, and where it was checked."""

    text: str
    citation_numbers: list[int] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    source: str = ""


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def collect_verified_facts(
    claim_verdicts: list[Any], citations: list[Any], hits: list[Any]
) -> list[VerifiedFact]:
    """The claims that passed Gate 2, de-duplicated, each tied to its source.

    ``claim_verdicts`` are Gate 2's per-claim verdicts; a claim's
    ``evidence_chunk_ids`` are mapped to the answer's citation numbers (the
    ``[n]`` the reader sees) and to the chunk's page.
    """
    page_by_chunk = {h.chunk_id: h.metadata.get("page") for h in hits}
    facts: list[VerifiedFact] = []
    seen: set[str] = set()
    for verdict in claim_verdicts:
        if verdict.verdict not in VERIFIED_VERDICTS:
            continue
        key = _norm(verdict.claim_text)
        if key in seen:
            continue
        seen.add(key)

        chunk_ids = list(getattr(verdict, "evidence_chunk_ids", None) or [])
        numbers: list[int] = []
        source = ""
        for citation in citations:
            if any(c in citation.contributing_chunk_ids for c in chunk_ids):
                if citation.number not in numbers:
                    numbers.append(citation.number)
                source = source or Path(str(citation.source_path)).name
        pages = sorted(
            {p for p in (page_by_chunk.get(c) for c in chunk_ids) if isinstance(p, int)}
        )
        facts.append(
            VerifiedFact(
                text=verdict.claim_text.strip(),
                citation_numbers=sorted(numbers),
                chunk_ids=chunk_ids,
                pages=pages,
                source=source,
            )
        )
    return facts


def _fact_line(fact: VerifiedFact) -> str:
    cites = "".join(f"[{n}]" for n in fact.citation_numbers)
    where = ""
    if fact.source:
        where = f" ({fact.source}" + (
            f", p. {', '.join(str(p) for p in fact.pages)}" if fact.pages else ""
        ) + ")"
    return f"- {fact.text}" + (f" {cites}" if cites else "") + where


def render_partial_answer(answer_text: str, facts: list[VerifiedFact]) -> str:
    """``answer_text`` unchanged, then the verified-facts list and the caution."""
    lines = "\n".join(_fact_line(f) for f in facts)
    return (
        f"{answer_text.rstrip()}\n\n{VERIFIED_FACTS_HEADING}\n{lines}\n\n{UNVERIFIED_NOTE}"
    )


_SECTION_RE = re.compile(
    rf"^\s*{re.escape(VERIFIED_FACTS_HEADING)}\s*$", re.MULTILINE
)


def split_verified_facts(output: str) -> tuple[str, str | None]:
    """``(answer_text, facts_section)``; the section is ``None`` if there is none."""
    match = _SECTION_RE.search(output or "")
    if match is None:
        return output, None
    body = output[: match.start()].rstrip()
    facts = output[match.end():].split(UNVERIFIED_NOTE)[0].strip()
    return body, facts
