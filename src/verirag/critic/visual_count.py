"""Count or read a value off a figure, by asking the vision model twice.

Measured on tier-2: of the 62 answerable questions still wrong under the best
gated configuration, 24 ask for something only visible in an image -- how many
arrows are in a logo, how many people wear glasses in a photo, how many bars
exceed a threshold. For 13 of the 15 that are genuinely image questions the answer
is in no stored chunk at all, because a caption DESCRIBES a figure rather than
enumerating its contents. Retrieval, verification and extraction changes cannot
reach them.

This asks the figure directly, and guards against the model's willingness to
agree with itself:

  * READ #1 sees the whole page at 200 dpi -- context, so "which figure" is clear.
  * READ #2 sees only that figure's own image, upscaled, so a small logo is large
    enough to count.
  * The two prompts are worded differently and NEITHER is shown the claim or any
    candidate answer.
  * A value is accepted only when both reads agree after normalisation.

Zero is a legal answer and the prompts say so: two of the target questions have a
gold answer of 0, and a model asked to count tends to invent something rather than
report none.

Deviation from the original design: the figure's bounding box is not stored, so
READ #2 uses the image file saved at ingest (``image_path``), which already IS the
figure crop -- the extracted raster, or the rendered crop for a vector chart --
upscaled by ``CROP_ZOOM``. If that file is gone, the page render is used for both
reads and they are then distinguished only by prompt wording, which is weaker; the
log records which happened.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from verirag.config import settings

logger = logging.getLogger(__name__)

PAGE_DPI = 200
PAGE_MAX_SIDE_PX = 2600
CROP_ZOOM = 3.0            # a logo occupying a small part of the page must be legible
CROP_MAX_SIDE_PX = 2000
MAX_PAGES = 3
IMAGE_TYPES = ("figure_caption", "image_caption")

# The question must be asking for something countable or readable off a picture.
_COUNT_RE = re.compile(
    r"\b(how many|number of|how much|count of)\b", re.IGNORECASE
)
_COLOUR_RE = re.compile(
    r"\b(colou?r(ed|s)?|blue|red|green|yellow|orange|purple|black|white|gr[ae]y|shaded)\b",
    re.IGNORECASE,
)
_READ_RE = re.compile(
    r"\b(in the (figure|photo|image|chart|diagram|logo|slide|picture)"
    r"|from the (figure|photo|image|chart|diagram|logo)"
    r"|shown in|depicted|illustrated|appears? in)\b",
    re.IGNORECASE,
)

# Two deliberately different wordings. Neither mentions any candidate answer.
_PROMPT_PAGE = """\
Look at this page image and answer the question from what is visible on it.

Question: {question}

Reply with exactly one line, and nothing else:
COUNT: <a whole number>
or
VALUE: <the value as printed>
or
UNREADABLE

Rules:
- Count or read only what you can actually see on this page.
- If there are none of the thing asked about, answer COUNT: 0. Zero is a correct
  answer and you must give it rather than guess a larger number.
- Do not estimate from bar heights or positions; a value must be printed to be read.
- If the page does not show what the question asks about, or it is too small to be
  sure, answer UNREADABLE."""

_PROMPT_CROP = """\
This image is one figure taken from a document.

Task: {question}

Answer on a single line using one of these three forms, with no other text:
COUNT: <number>
VALUE: <printed value>
UNREADABLE

Guidance:
- Base the answer solely on this image.
- A total of none is reported as COUNT: 0 -- do not round up to one.
- Report a printed value only; never infer one from a shape's size.
- Answer UNREADABLE if this figure does not contain what was asked, or detail is
  lost at this resolution."""


@dataclass
class CountRead:
    """One recorded read, for the pipeline log."""

    source: str          # "page <file> p<n>" or "crop <file>"
    prompt_variant: str  # "page" | "crop"
    raw_reply: str
    parsed: str | None   # normalised value, or None for UNREADABLE/error


@dataclass
class CountResult:
    """An agreed reading of a figure, or a record of why there is none."""

    value: str | None
    page: int | None
    source_path: str | None
    reads: list[CountRead] = field(default_factory=list)
    agreed: bool = False
    reason: str = ""


# (sha256 of image bytes, prompt variant, question) -> raw reply
_CACHE: dict[tuple[str, str, str], str] = {}


def should_count(query: str, hits: list) -> bool:
    """True if this question asks for something countable in a retrieved figure."""
    if not settings.visual_count:
        return False
    text = query or ""
    asks = bool(_COUNT_RE.search(text)) or (
        bool(_COLOUR_RE.search(text)) and bool(_READ_RE.search(text))
    ) or bool(_READ_RE.search(text))
    if not asks:
        return False
    return any(
        str((getattr(h, "metadata", None) or {}).get("content_type")) in IMAGE_TYPES
        for h in hits
    )


def figure_targets(hits: list, limit: int = MAX_PAGES) -> list:
    """The top retrieved figure chunks, at most one per page, highest ranked first."""
    out, seen = [], set()
    for hit in hits:
        meta = getattr(hit, "metadata", None) or {}
        if str(meta.get("content_type")) not in IMAGE_TYPES:
            continue
        key = (str(meta.get("source_path")), meta.get("page"))
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
        if len(out) >= limit:
            break
    return out


def render_page(source_path: str, page: int) -> bytes | None:
    """The whole page as a PNG at ``PAGE_DPI``."""
    try:
        import pymupdf

        with pymupdf.open(source_path) as pdf:
            if not 1 <= page <= pdf.page_count:
                return None
            pg = pdf[page - 1]
            scale = min(PAGE_DPI / 72,
                        PAGE_MAX_SIDE_PX / max(pg.rect.width, pg.rect.height))
            return pg.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                                 alpha=False).tobytes("png")
    except Exception as exc:  # noqa: BLE001 - a bad PDF is not fatal
        logger.debug("Page render failed (%s p%s): %s", source_path, page, exc)
        return None


def upscaled_crop(image_path: str) -> bytes | None:
    """The figure's own image, enlarged so small marks can be counted."""
    try:
        import pymupdf

        data = Path(str(image_path)).read_bytes()
        if not data:
            return None
        pix = pymupdf.Pixmap(data)
        if pix.colorspace is None or pix.n - pix.alpha not in (1, 3):
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
        zoom = min(CROP_ZOOM, CROP_MAX_SIDE_PX / max(pix.width, pix.height, 1))
        if zoom <= 1.0:
            return pix.tobytes("png")
        big = pymupdf.Pixmap(pix, 0)  # copy
        return _scale_png(big, zoom)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Crop upscale failed (%s): %s", image_path, exc)
        return None


def _scale_png(pix, zoom: float) -> bytes:
    """``pix`` enlarged by ``zoom``, via a one-page PDF so PyMuPDF does the resampling."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=pix.width, height=pix.height)
    page.insert_image(page.rect, pixmap=pix)
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).tobytes("png")


def _mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def ask(data: bytes, question: str, variant: str) -> str:
    """One blind vision call; returns the raw reply or ``error:<reason>``."""
    from verirag.loaders.captioning import _client

    prompt = (_PROMPT_PAGE if variant == "page" else _PROMPT_CROP).format(question=question)
    url = f"data:{_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"
    try:
        response = _client().chat.completions.create(
            model=settings.deepseek_vision_model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}},
            ]}],
            temperature=0.0,
            max_tokens=settings.gate2_visual_check_max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 - never let a vision error break Gate 2
        return f"error:{type(exc).__name__}"


def parse_read(reply: str) -> str | None:
    """The normalised value in ``reply``, or None for UNREADABLE and anything unclear.

    Fails closed: only an explicit COUNT or VALUE line yields a value, so a chatty
    or malformed reply cannot certify anything.
    """
    if not reply or reply.startswith("error:"):
        return None
    for line in reply.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("UNREADABLE"):
            return None
        if upper.startswith(("COUNT:", "VALUE:")):
            raw = line.split(":", 1)[1].strip()
            return normalise_value(raw) if raw else None
    return None


def normalise_value(raw: str) -> str | None:
    """A comparable form of a read value: the number if there is one, else the text."""
    from verirag.critic.gate2_groundedness import parse_number

    raw = (raw or "").strip().strip(".")
    if not raw:
        return None
    match = re.search(r"-?\d[\d,.]*", raw)
    if match:
        value = parse_number(match.group(0).rstrip(".,"))
        if value is not None:
            return str(int(value)) if float(value).is_integer() else f"{value:g}"
    return " ".join(raw.lower().split()) or None


@dataclass
class VisualCounter:
    """Per-question budget holder for the two-read counting protocol."""

    budget: int = 0
    reads: list[CountRead] = field(default_factory=list)
    used: int = 0

    def __post_init__(self) -> None:
        if not self.budget:
            self.budget = settings.visual_count_calls_per_question

    def _one(self, data: bytes, question: str, variant: str, source: str) -> str | None:
        key = (hashlib.sha256(data).hexdigest(), variant, question)
        if key in _CACHE:
            reply = _CACHE[key]
        else:
            if self.used >= self.budget:
                self.reads.append(CountRead(source, variant, "error:budget_exhausted", None))
                return None
            self.used += 1
            reply = ask(data, question, variant)
            if not reply.startswith("error:"):
                _CACHE[key] = reply
        parsed = parse_read(reply)
        self.reads.append(CountRead(source, variant, reply[:200], parsed))
        return parsed

    def count(self, question: str, hits: list) -> CountResult:
        """Read the top figures twice each; return the first agreed value.

        Agreement means both reads of the SAME figure produced the same normalised
        value. Disagreement, UNREADABLE, a missing image or an exhausted budget all
        leave ``value`` None, and the caller changes nothing.
        """
        result = CountResult(value=None, page=None, source_path=None)
        targets = figure_targets(hits)
        if not targets:
            result.reason = "no figure chunk retrieved"
            return result
        for hit in targets:
            meta = getattr(hit, "metadata", None) or {}
            source, page = str(meta.get("source_path") or ""), meta.get("page")
            name = Path(source).name
            page_png = render_page(source, page) if isinstance(page, int) else None
            crop_png = upscaled_crop(meta["image_path"]) if meta.get("image_path") else None
            if crop_png is None and page_png is None:
                continue
            # Two different images where possible; the log says when it was not.
            first = self._one(page_png or crop_png, question, "page", f"page {name} p{page}")
            second = self._one(crop_png or page_png, question, "crop",
                               f"crop {Path(str(meta.get('image_path') or name)).name}"
                               if crop_png else f"page {name} p{page} (no crop available)")
            result.reads = list(self.reads)
            if first is not None and first == second:
                result.value, result.page, result.source_path = first, page, source
                result.agreed = True
                result.reason = "both reads agree"
                logger.info("Visual count agreed on %r for %r (%s p%s)",
                            first, question[:70], name, page)
                return result
            logger.info("Visual count disagreed (%r vs %r) for %r on %s p%s",
                        first, second, question[:70], name, page)
        result.reads = list(self.reads)
        result.reason = result.reason or "reads did not agree"
        return result
