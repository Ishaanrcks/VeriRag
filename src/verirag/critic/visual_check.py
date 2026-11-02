"""Verify a caption-sourced claim against the FIGURE ITSELF, not its caption.

Gate 2 refuses to certify a claim whose only support is a ``figure_caption`` chunk:
that caption is one model's description, and certifying against it would mean a
model vouching for its own output (``caption_only_uncertified``). The rule is
right, but it was throwing away correct answers -- measured on the tier-2
benchmark, Gate 2 rejected claims that quoted the gold answer verbatim ("the pie
chart labels iOS 9 at 51%" against a gold answer of 51%), because a chart's
numbers often exist nowhere but the caption.

This module keeps the principle and changes the evidence: the caption still never
certifies anything, but the PIXELS may. The check is ADVERSARIAL in two steps, and
only numeric claims are eligible:

  1. BLIND READ -- the figure and the QUESTION are sent to the vision model, never
     the claim. It reports the value it can read, or UNREADABLE.
  2. COMPARE IN CODE -- that value is matched against the claim's numbers, after
     normalisation. Only a MATCH certifies, as ``visually_verified``; MISMATCH and
     UNREADABLE keep today's rejection.

The first version asked "does this image show <claim>?" and was measured to answer
YES on 20 of 22 checks: a leading question buys assent, not verification. Reading
the value independently and comparing it outside the model is what lets the check
disagree.

This is a second look at the same evidence by the same model family, so it is
weaker than entailment against document-extracted text. It is deliberately bounded:
at most ``gate2_visual_checks_per_answer`` calls per answer, a cache so a repeated
(image, claim) pair costs nothing, temperature 0, and every check is logged.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from verirag.config import settings

logger = logging.getLogger(__name__)

VisualVerdict = Literal["MATCH", "MISMATCH", "UNREADABLE"]

# Rendering DPI for the page fallback, when the figure's own image file is gone.
PAGE_RENDER_DPI = 160
PAGE_RENDER_MAX_SIDE_PX = 2200

# Call 1 is BLIND: the model is shown the figure and the QUESTION only, never the
# claim. Asking "does this image show X?" was measured to answer YES 20 times out of
# 22 -- a leading question gets assent, not verification. Reading the value off the
# figure independently and comparing it to the claim IN CODE is what makes the check
# able to disagree.
_READ_PROMPT = """\
Read the value this question asks for directly off the image.

Question: {question}

Reply with exactly one line, and nothing else:
VALUE: <the value, as it appears in the image>
or
UNREADABLE

Rules:
- Report only what is printed or plainly legible in the image. Do not estimate from
  bar heights or positions, and do not use outside knowledge.
- If the image does not contain the value the question asks for, or it is not
  legible, reply UNREADABLE.
- Give the value alone -- a number, a percentage, a label -- with no explanation."""


@dataclass
class VisualCheck:
    """One recorded check, for the pipeline log."""

    claim: str
    image: str           # where the pixels came from
    verdict: str         # MATCH | MISMATCH | UNREADABLE | error:<reason> | cached:<v>
    chunk_id: str = ""
    reported: str = ""   # what call 1 read off the figure, before comparison


# (sha256 of image bytes, claim) -> verdict. Module-level so a figure re-checked
# across regeneration attempts, or across questions in a benchmark run, is free.
# (sha256 of image bytes, question) -> the value call 1 read off the figure.
_CACHE: dict[tuple[str, str], str] = {}


def _render_page(source_path: str, page: int) -> bytes | None:
    """The claim's page as a PNG, for when the figure's own image file is missing."""
    try:
        import pymupdf

        with pymupdf.open(source_path) as pdf:
            if not 1 <= page <= pdf.page_count:
                return None
            pg = pdf[page - 1]
            scale = min(PAGE_RENDER_DPI / 72,
                        PAGE_RENDER_MAX_SIDE_PX / max(pg.rect.width, pg.rect.height))
            return pg.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")
    except Exception as exc:  # noqa: BLE001 - a missing/corrupt PDF is not fatal
        logger.debug("Page render failed (%s p%s): %s", source_path, page, exc)
        return None


def resolve_image(hit) -> tuple[bytes, str] | None:
    """``(png_or_original_bytes, provenance)`` for a caption chunk's figure.

    Prefers the image saved at ingest (``image_path``; for a vector chart that is
    the rendered crop). Falls back to rendering the caption's page out of the
    source PDF, which is why a store moved between machines still verifies.
    """
    metadata = getattr(hit, "metadata", None) or {}
    path = metadata.get("image_path")
    if path:
        try:
            data = Path(str(path)).read_bytes()
            if data:
                return data, str(path)
        except OSError as exc:
            logger.debug("Stored figure image unreadable (%s): %s", path, exc)

    source, page = metadata.get("source_path"), metadata.get("page")
    if source and isinstance(page, int):
        rendered = _render_page(str(source), page)
        if rendered:
            return rendered, f"page render {Path(str(source)).name} p{page}"
    return None


def _mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _read_value(data: bytes, question: str) -> str:
    """Call 1, blind: what value does the FIGURE give for ``question``?

    Returns the reported value, ``UNREADABLE``, or ``error:<reason>``. The claim is
    never sent, so the model cannot simply agree with it.
    """
    from verirag.loaders.captioning import _client

    url = f"data:{_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"
    try:
        response = _client().chat.completions.create(
            model=settings.deepseek_vision_model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _READ_PROMPT.format(question=question)},
                {"type": "image_url", "image_url": {"url": url}},
            ]}],
            temperature=0.0,
            max_tokens=settings.gate2_visual_check_max_tokens,
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 - never let a vision error break Gate 2
        return f"error:{type(exc).__name__}"
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("VALUE:"):
            return line.split(":", 1)[1].strip() or "UNREADABLE"
        if line.upper().startswith("UNREADABLE"):
            return "UNREADABLE"
    return "UNREADABLE"


def values_agree(reported: str, claim: str) -> bool:
    """True if the figure's reported value appears among the claim's numbers.

    Compared on normalised numbers so "51%", "51 percent" and "51" agree, and
    "1.3 million" agrees with "1300000". A reported value with no number in it
    never agrees -- a label match is too weak to certify a numeric claim.
    """
    from verirag.critic.gate2_groundedness import normalised_numbers

    reported_numbers = normalised_numbers(reported)
    if not reported_numbers:
        return False
    return bool(reported_numbers & normalised_numbers(claim))


@dataclass
class VisualChecker:
    """Per-answer budget holder. One instance per ``check_groundedness`` call."""

    query: str = ""
    budget: int = 0
    checks: list[VisualCheck] = field(default_factory=list)
    # Vision calls actually made. Counted explicitly rather than derived from
    # ``checks``, which also records entries where NO call happened (cache hit, no
    # image, budget exhausted) and would otherwise consume the budget twice over.
    used: int = 0

    def __post_init__(self) -> None:
        if not self.budget:
            self.budget = settings.gate2_visual_checks_per_answer

    def check(self, claim: str, hit) -> VisualVerdict | None:
        """MATCH / MISMATCH / UNREADABLE for ``claim`` against ``hit``'s figure.

        ``None`` means no judgement was possible (flag off, no numbers in the claim,
        no question, no image, no budget, or a vision error) and the caller keeps its
        existing rejection. Only MATCH certifies.
        """
        if not settings.gate2_visual_verify:
            return None
        if not self.query:
            return None
        # Numeric claims only: the comparison is on numbers, so a claim with none
        # cannot be checked this way.
        from verirag.critic.gate2_groundedness import normalised_numbers

        if not normalised_numbers(claim):
            return None

        resolved = resolve_image(hit)
        if resolved is None:
            self.checks.append(VisualCheck(claim[:120], "none", "error:no_image",
                                           getattr(hit, "chunk_id", "")))
            return None
        data, provenance = resolved
        # Keyed on the QUESTION, not the claim: call 1 never sees the claim, so the
        # same figure and question always produce the same reading.
        key = (hashlib.sha256(data).hexdigest(), self.query)
        if key in _CACHE:
            reported = _CACHE[key]
            verdict = _compare(reported, claim)
            self.checks.append(VisualCheck(claim[:120], provenance, f"cached:{verdict}",
                                           getattr(hit, "chunk_id", ""), reported))
            return verdict
        if self.used >= self.budget:
            self.checks.append(VisualCheck(claim[:120], provenance, "error:budget_exhausted",
                                           getattr(hit, "chunk_id", "")))
            return None

        self.used += 1
        reported = _read_value(data, self.query)
        if reported.startswith("error:"):
            self.checks.append(VisualCheck(claim[:120], provenance, reported,
                                           getattr(hit, "chunk_id", "")))
            logger.warning("Visual read failed (%s) for %r", reported, self.query[:80])
            return None
        _CACHE[key] = reported
        verdict = _compare(reported, claim)
        self.checks.append(VisualCheck(claim[:120], provenance, verdict,
                                       getattr(hit, "chunk_id", ""), reported))
        logger.info("Visual read %r -> %s against claim %r", reported[:40], verdict, claim[:70])
        return verdict


def _compare(reported: str, claim: str) -> VisualVerdict:
    if reported.upper().startswith("UNREADABLE"):
        return "UNREADABLE"
    return "MATCH" if values_agree(reported, claim) else "MISMATCH"
