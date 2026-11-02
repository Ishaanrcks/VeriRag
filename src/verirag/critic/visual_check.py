"""Verify a caption-sourced claim against the FIGURE ITSELF, not its caption.

Gate 2 refuses to certify a claim whose only support is a ``figure_caption`` chunk:
that caption is one model's description, and certifying against it would mean a
model vouching for its own output (``caption_only_uncertified``). The rule is
right, but it was throwing away correct answers -- measured on the tier-2
benchmark, Gate 2 rejected claims that quoted the gold answer verbatim ("the pie
chart labels iOS 9 at 51%" against a gold answer of 51%), because a chart's
numbers often exist nowhere but the caption.

This module keeps the principle and changes the evidence: the caption still never
certifies anything, but the PIXELS may. When a claim would be
``caption_only_uncertified``, the figure's own image is sent to the vision model
with the claim and a strict three-way question. Only an unambiguous YES certifies,
as ``visually_verified``; NO and CANNOT_TELL keep today's rejection.

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

VisualVerdict = Literal["YES", "NO", "CANNOT_TELL"]

# Rendering DPI for the page fallback, when the figure's own image file is gone.
PAGE_RENDER_DPI = 160
PAGE_RENDER_MAX_SIDE_PX = 2200

_PROMPT = """\
You are verifying one factual claim against the image shown.

Claim: {claim}

Answer with exactly one word, and nothing else:
YES          -- the image itself clearly shows the claim to be true. Every number,
                label and relationship the claim asserts is legible in the image.
NO           -- the image shows something that contradicts the claim.
CANNOT_TELL  -- the image does not settle it: the detail is illegible, absent, or
                you would have to assume something the image does not show.

Judge ONLY what is visible in this image. Do not use outside knowledge. If the
claim states a number, that number must actually be readable in the image. When in
doubt, answer CANNOT_TELL."""


@dataclass
class VisualCheck:
    """One recorded check, for the pipeline log."""

    claim: str
    image: str           # where the pixels came from
    verdict: str         # YES | NO | CANNOT_TELL | error:<reason> | cached:<verdict>
    chunk_id: str = ""


# (sha256 of image bytes, claim) -> verdict. Module-level so a figure re-checked
# across regeneration attempts, or across questions in a benchmark run, is free.
_CACHE: dict[tuple[str, str], VisualVerdict] = {}


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


def _ask(data: bytes, claim: str) -> VisualVerdict | str:
    """One vision call; returns a verdict or ``error:<reason>``."""
    from verirag.loaders.captioning import _client

    url = f"data:{_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"
    try:
        response = _client().chat.completions.create(
            model=settings.deepseek_vision_model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": _PROMPT.format(claim=claim)},
                {"type": "image_url", "image_url": {"url": url}},
            ]}],
            temperature=0.0,
            max_tokens=settings.gate2_visual_check_max_tokens,
        )
        text = (response.choices[0].message.content or "").strip().upper()
    except Exception as exc:  # noqa: BLE001 - never let a vision error break Gate 2
        return f"error:{type(exc).__name__}"

    # Strict parse, failing closed: anything that is not an unambiguous YES is a
    # refusal to certify. "NO" is checked first because "CANNOT_TELL" contains no
    # "YES" but a chatty reply might contain both words.
    if text.startswith("CANNOT_TELL") or "CANNOT_TELL" in text:
        return "CANNOT_TELL"
    if text.startswith("NO"):
        return "NO"
    if text.startswith("YES"):
        return "YES"
    logger.warning("Unparseable visual verdict %r; treating as CANNOT_TELL.", text[:60])
    return "CANNOT_TELL"


@dataclass
class VisualChecker:
    """Per-answer budget holder. One instance per ``check_groundedness`` call."""

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
        """YES/NO/CANNOT_TELL for ``claim`` against ``hit``'s figure, or None.

        ``None`` means no judgement was possible (no image, no budget left, or the
        vision call failed) and the caller must keep its existing rejection.
        """
        if not settings.gate2_visual_check_enabled:
            return None
        resolved = resolve_image(hit)
        if resolved is None:
            self.checks.append(VisualCheck(claim[:120], "none", "error:no_image",
                                           getattr(hit, "chunk_id", "")))
            return None
        data, provenance = resolved
        key = (hashlib.sha256(data).hexdigest(), claim)
        if key in _CACHE:
            verdict = _CACHE[key]
            self.checks.append(VisualCheck(claim[:120], provenance, f"cached:{verdict}",
                                           getattr(hit, "chunk_id", "")))
            return verdict
        if self.used >= self.budget:
            self.checks.append(VisualCheck(claim[:120], provenance, "error:budget_exhausted",
                                           getattr(hit, "chunk_id", "")))
            return None

        self.used += 1
        verdict = _ask(data, claim)
        self.checks.append(VisualCheck(claim[:120], provenance, verdict,
                                       getattr(hit, "chunk_id", "")))
        if verdict in ("YES", "NO", "CANNOT_TELL"):
            _CACHE[key] = verdict  # type: ignore[assignment]
            logger.info("Visual check %s for claim %r against %s", verdict, claim[:80], provenance)
            return verdict  # type: ignore[return-value]
        logger.warning("Visual check failed (%s) for claim %r", verdict, claim[:80])
        return None
