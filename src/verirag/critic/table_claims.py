"""Verify a numeric claim against a table chunk by cell-level presence.

NLI is a poor judge of a claim like "West Germany is at 29% for Both" against a
markdown table: the premise is pipes and numbers, and entailment scored a true
claim 0.0057-0.49. The right question is a lookup -- is there a cell holding
that number, in the row and column the claim names? -- so this module answers
it directly, the way ``structural`` claims are checked by finding a quoted title.

A claim is verified only if EVERY number it states is paired with a cell that:

  * holds exactly that number (``29`` == ``29%`` == ``29.0``; ``30`` != ``29``);
  * sits in a row whose label (the row's first non-empty cell) is named in the
    claim; and
  * sits under a column header that is named in the claim.

So a number that appears only in another row or column does not verify, and a
claim that names no row or no column does not verify. A claim that compares,
negates or aggregates ("more than", "not", "total") is left to NLI, because a
correct cell does not make such a claim true.

Typeset tables often leave a value one column off its header (a right-aligned
figure under a spacer column). A value under a BLANK header is therefore paired
with the nearest non-blank header on a side whose cell in that row is empty, but
only when exactly one side qualifies.

This only ever ADDS a way to verify; a claim it cannot verify falls through to
the normal NLI check unchanged. It certifies the (row, column, value) fact, not
any other wording in the claim -- the guard against comparatives and negation is
what keeps that from being a loophole.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_TOKEN_RE = re.compile(r"\w{2,}")
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")
_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
# Words after which a correct cell value no longer makes the claim true.
_UNSAFE_RE = re.compile(
    r"\b(?:not|no|never|none|neither|than|more|less|fewer|higher|lower|greater|"
    r"smaller|larger|twice|double|half|majority|minority|most|least|only|all|"
    r"every|total|sum|average|combined|difference|increase[ds]?|decrease[ds]?|"
    r"rose|fell|versus|vs|compared|exceed(?:s|ed)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CellMatch:
    """One claim number paired with the table cell that backs it."""

    number: str
    row_label: str
    column_header: str
    shifted: bool  # the header was inferred across a blank header cell


def _stem(token: str) -> str:
    return token[:6]


def _normalise(text: str) -> str:
    """Lower-case; drop the dots inside acronyms ("U.S." -> "us")."""
    return re.sub(r"(?<=[A-Za-z])\.(?=[A-Za-z])", "", text).lower()


def _stems(text: str, drop_parenthetical: bool = False) -> set[str]:
    if drop_parenthetical:
        stripped = re.sub(r"\([^)]*\)", " ", text)
        text = stripped if _TOKEN_RE.search(stripped) else text
    return {_stem(t) for t in _TOKEN_RE.findall(_normalise(text))}


def _as_decimal(text: str) -> Decimal | None:
    cleaned = text.strip().replace(",", "").replace("$", "").rstrip("%").strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned):
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:  # pragma: no cover - regex already guards this
        return None


def parse_markdown_table(text: str) -> list[list[str]] | None:
    """Rows of cells from a GFM markdown table chunk, or ``None`` if it is not one."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 3 or not lines[0].lstrip().startswith("|") or not _SEPARATOR_RE.match(lines[1]):
        return None
    rows: list[list[str]] = []
    for i, line in enumerate(lines):
        if i == 1:
            continue
        body = line.strip()
        body = body[1:] if body.startswith("|") else body
        body = body[:-1] if body.endswith("|") and not body.endswith("\\|") else body
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", body)]
        rows.append(cells)
    return rows


def _header_for(header: list[str], row: list[str], col: int) -> tuple[str, bool] | None:
    """The header that names column ``col`` for ``row``, and whether it was inferred."""
    if col < len(header) and header[col].strip():
        return header[col], False
    candidates = []
    for step in (-1, 1):
        c = col + step
        while 0 <= c < len(header) and not header[c].strip():
            c += step
        if 0 <= c < len(header) and (c >= len(row) or not row[c].strip()):
            candidates.append(header[c])
    return (candidates[0], True) if len(candidates) == 1 else None


def verify_numeric_claim(claim: str, table_text: str) -> list[CellMatch] | None:
    """Pair every number in ``claim`` with a named cell of ``table_text``.

    Returns the matches, or ``None`` if the claim cannot be verified this way
    (no numbers, an unsafe comparative/negation, an unparseable table, or any
    number without a correctly-paired cell).
    """
    numbers = _NUMBER_RE.findall(claim)
    if not numbers or _UNSAFE_RE.search(claim):
        return None
    rows = parse_markdown_table(table_text)
    if rows is None or len(rows) < 2:
        return None
    header, body = rows[0], rows[1:]
    claim_stems = _stems(claim)

    matches: list[CellMatch] = []
    for number in numbers:
        wanted = _as_decimal(number)
        if wanted is None:
            return None
        found: CellMatch | None = None
        for row in body:
            label_idx = next((i for i, c in enumerate(row) if c.strip()), None)
            if label_idx is None or _as_decimal(row[label_idx]) is not None:
                continue  # blank row, or the first cell is itself a number
            label_cell = row[label_idx]
            label = _stems(label_cell)
            if not label or not label <= claim_stems:
                continue
            for col, cell in enumerate(row):
                if col == label_idx or _as_decimal(cell) != wanted:
                    continue
                named = _header_for(header, row, col)
                if named is None:
                    continue
                head, shifted = named
                head_stems = _stems(head, drop_parenthetical=True)
                if head_stems and head_stems <= claim_stems:
                    found = CellMatch(number, label_cell, head, shifted)
                    break
            if found:
                break
        if found is None:
            return None
        matches.append(found)
    return matches
