"""Find charts that a PDF draws as VECTOR graphics, and render them to images.

``load_pdf`` captions the raster images embedded in a page. A chart drawn as vector
paths (bars, lines, pie slices written as PDF drawing operators) has no embedded
image, so that logic never sees it: the chart is silently dropped and only its
scrambled text labels survive. This module finds such charts and renders each to an
image, so it can enter the same image -> caption flow as an ordinary figure.

The rule is the "strict" one validated by scanning 4,616 pages of the 103
MMLongBench-Doc corpus documents and checking samples by eye (24 of 25 flagged
regions were real data charts; the one miss was an infographic). It is deliberately
conservative: a looser rule also flagged banded financial tables, shaded callout
boxes and calligraphy, and found only about one real chart in three of the extras.
The cost of that conservatism is recall (roughly a third of the remaining
chart-evidence pages still hold a chart this rule does not catch).

A page region qualifies when ALL of these hold:

  * it is a cluster of drawing primitives at least 120 x 70 pt, not a whole-page frame;
  * it has either several distinct fill colours across at least five shapes (bars,
    pie slices, stacked bars) or real line/curve geometry (12 or more slanted line
    segments or curves: line plots, density plots, dot-and-line charts);
  * it has at most 5 hairline rules (table borders and row bands come with dozens). A
    hairline is a path under 2 pt thick in either direction AND with a non-zero area,
    i.e. a thin filled or stroked rectangle, which is how the banded tables in the
    corpus draw their rules. A plain zero-width line has no area and is not counted
    (the same behaviour as in the scan that validated the rule);
  * it carries at least 3 numeric labels and at most 150 words (a chart is mostly
    marks and numbers; a text box or code listing is mostly words);
  * no image that ``load_pdf`` will already caption overlaps it.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

import pymupdf

logger = logging.getLogger(__name__)

MIN_WIDTH_PT = 120
MIN_HEIGHT_PT = 70
MAX_PAGE_FRACTION = 0.85     # a region covering the page is a background or frame
MAX_PATHS = 20_000           # a page with more primitives than this is skipped
MAX_PER_PAGE = 6             # safety cap; the corpus maximum is 4
RASTER_OVERLAP = 0.3         # share of the region covered by a captioned raster
MIN_FILLS, MIN_FILLED_PATHS = 3, 5
MIN_GEOMETRY = 12
MAX_HAIRLINES = 5
MIN_NUMERIC_WORDS = 3
MAX_WORDS = 150

RENDER_DPI = 200             # high enough that a small chart still clears the size floor
RENDER_PAD_PT = 8            # takes in axis labels and legends just outside the marks
RENDER_MAX_SIDE_PX = 2400

_NUMERIC = re.compile(r"^[\(\-$]?\d[\d.,]*%?\)?$")


@dataclass(frozen=True)
class VectorRegion:
    """A chart-like cluster of vector drawing, with the features that qualified it."""

    rect: pymupdf.Rect
    features: dict


def _is_white(color) -> bool:
    return color is not None and all(v > 0.95 for v in color)


def _is_black(color) -> bool:
    return color is not None and all(v < 0.05 for v in color)


def describe_region(page: "pymupdf.Page", region: pymupdf.Rect, drawings: list[dict]) -> dict:
    """Counts of what is drawn inside ``region``."""
    inside = [
        d for d in drawings
        if region.intersects(d["rect"])
        and (d["rect"] & region).get_area() >= 0.5 * max(d["rect"].get_area(), 1e-6)
    ]
    fills = Counter(
        tuple(round(v, 2) for v in d["fill"])
        for d in inside
        if d.get("fill") and not _is_white(d["fill"]) and not _is_black(d["fill"])
    )
    kinds = Counter(item[0] for d in inside for item in d["items"])
    hairlines = 0
    slanted = 0
    for d in inside:
        if min(d["rect"].width, d["rect"].height) < 2.0:
            hairlines += 1
        for item in d["items"]:
            if item[0] == "l":
                p, q = item[1], item[2]
                if abs(p.x - q.x) > 1.5 and abs(p.y - q.y) > 1.5:
                    slanted += 1
    words = page.get_text("words", clip=region)
    return dict(
        paths=len(inside),
        distinct_fills=len(fills),
        curves=kinds.get("c", 0),
        slanted_lines=slanted,
        hairlines=hairlines,
        words=len(words),
        numeric_words=sum(1 for w in words if _NUMERIC.match(w[4])),
    )


def is_chart_like(f: dict) -> bool:
    """The strict rule (see the module docstring), on ``describe_region`` features."""
    marks = (
        (f["distinct_fills"] >= MIN_FILLS and f["paths"] >= MIN_FILLED_PATHS)
        or f["slanted_lines"] >= MIN_GEOMETRY
        or f["curves"] >= MIN_GEOMETRY
    )
    return (
        marks
        and f["hairlines"] <= MAX_HAIRLINES
        and f["numeric_words"] >= MIN_NUMERIC_WORDS
        and f["words"] <= MAX_WORDS
    )


def find_vector_charts(page: "pymupdf.Page", *, min_raster_px: int) -> list[VectorRegion]:
    """Chart-like vector regions on ``page`` that no captioned raster already covers.

    ``min_raster_px`` is the pipeline's raster size floor: an image smaller than that
    is NOT captioned by ``load_pdf``, so it does not count as covering the chart.
    Returns at most ``MAX_PER_PAGE`` regions, top to bottom. Never raises: a page
    that cannot be analysed simply yields no regions.
    """
    try:
        drawings = page.get_drawings()
        if not drawings or len(drawings) > MAX_PATHS:
            return []
        captioned = [
            pymupdf.Rect(info["bbox"])
            for info in page.get_image_info(xrefs=True)
            if min(info["width"], info["height"]) >= min_raster_px
        ]
        page_area = page.rect.get_area()
        found: list[VectorRegion] = []
        for region in page.cluster_drawings(drawings=drawings):
            if (
                region.width < MIN_WIDTH_PT
                or region.height < MIN_HEIGHT_PT
                or region.get_area() > MAX_PAGE_FRACTION * page_area
            ):
                continue
            if any((box & region).get_area() >= RASTER_OVERLAP * region.get_area()
                   for box in captioned):
                continue
            features = describe_region(page, region, drawings)
            if is_chart_like(features):
                found.append(VectorRegion(region, features))
    except Exception as exc:  # noqa: BLE001 - never abort a document load
        logger.warning("Vector chart detection failed on page %s: %s", page.number + 1, exc)
        return []
    found.sort(key=lambda r: (r.rect.y0, r.rect.x0))
    return found[:MAX_PER_PAGE]


def render_region(page: "pymupdf.Page", rect: pymupdf.Rect) -> tuple[bytes, int, int]:
    """PNG bytes of ``rect`` (padded, clipped to the page) and its pixel size."""
    clip = pymupdf.Rect(
        max(rect.x0 - RENDER_PAD_PT, page.rect.x0), max(rect.y0 - RENDER_PAD_PT, page.rect.y0),
        min(rect.x1 + RENDER_PAD_PT, page.rect.x1), min(rect.y1 + RENDER_PAD_PT, page.rect.y1),
    )
    scale = RENDER_DPI / 72
    scale = min(scale, RENDER_MAX_SIDE_PX / max(clip.width, clip.height))
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False)
    return pixmap.tobytes("png"), pixmap.width, pixmap.height
