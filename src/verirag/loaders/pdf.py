"""PDF loader: per-page text, extracted tables, and captioned figures (PyMuPDF).

Per page, the loader emits (in this order):
  * ``content_type="table"`` Documents -- each table found by
    ``page.find_tables()`` serialized to GitHub-Flavour Markdown. Tables are
    extracted directly from the document structure, so they are document-
    derived evidence, not model output.
  * ``content_type="text"`` Documents -- the page's text via
    ``page.get_text()`` (one per page that has text).
  * ``content_type="figure_caption"`` Documents -- a vision-model caption for
    each embedded image large enough to be a real figure. The raw image bytes
    are ALSO saved to ``IMAGES_DIR`` and the path recorded in the chunk's
    ``image_path`` metadata, so the UI can show the real figure later.

De-duplicating a table's cell text out of the same page's raw-text chunk is
deliberately NOT done: the serialized markdown and ``get_text()`` output differ
too much (PyMuPDF splits cell tokens across lines) for a reliable substring
strip, so a table's numbers may appear in both its table chunk and the page's
text chunk. This is a known, minor redundancy that does not affect retrieval
correctness -- more grounding evidence, not less.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # the modern import name for PyMuPDF (aka ``fitz``)
from langchain_core.documents import Document

from verirag.config import settings
from verirag.loaders.base import compute_doc_id, make_document
from verirag.loaders.captioning import caption_image_with_status

logger = logging.getLogger(__name__)

# Persisted figure images live here (see config.images_dir). Created at import
# so the directory exists before the first ingest; a failure here must not stop
# the module from importing (the UI imports IMAGES_DIR only for a path guard),
# so it is best-effort and the write path re-creates it defensively.
IMAGES_DIR = Path(settings.images_dir)
try:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
except OSError as exc:  # pragma: no cover - only on a read-only filesystem
    logger.warning("Could not create images dir %s: %s", IMAGES_DIR, exc)

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_cell(cell: object) -> str:
    """One table cell as a single-line, pipe-safe markdown token.

    Extracted cells routinely contain embedded newlines (PyMuPDF breaks a cell
    across physical lines), which would break a GFM table row, so newlines and
    runs of whitespace collapse to a single space. A literal ``|`` is escaped
    so it is not read as a column separator.
    """
    text = "" if cell is None else str(cell)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.replace("|", r"\|")


def _table_to_markdown(rows: list[list]) -> str | None:
    """Serialize a 2D list of cell strings to a GitHub-Flavour Markdown table.

    Returns ``None`` for tables with fewer than 2 rows or fewer than 2 columns
    (likely false positives or single-value blocks). Every row is padded to the
    widest row's column count so the table stays valid GFM even when PyMuPDF
    reports ragged rows.
    """
    if len(rows) < 2:
        return None
    width = max((len(row) for row in rows), default=0)
    if width < 2:
        return None

    def _fmt(row: list) -> str:
        cells = [
            _normalize_cell(row[i]) if i < len(row) else "" for i in range(width)
        ]
        return "| " + " | ".join(cells) + " |"

    header = _fmt(rows[0])
    separator = "| " + " | ".join(["---"] * width) + " |"
    body = [_fmt(row) for row in rows[1:]]
    return "\n".join([header, separator, *body])


# A table PyMuPDF finds by ruling lines but whose body has no row rules comes
# back as a header plus ONE body row, each cell holding every entry joined by
# newlines (measured on an unruled attendance table: 14 directors in one cell).
# That signature -- at most this many rows, yet some cell with at least this many
# lines -- triggers a retry with PyMuPDF's whitespace-based "text" strategy.
_COLLAPSE_MAX_ROWS = 4
_COLLAPSE_MIN_CELL_LINES = 6
# Guards on accepting a text-strategy result in place of a collapsed one. The
# text strategy can also shred a page's prose, so a replacement must overlap the
# original, keep (almost) all of its words, be mostly filled, and not balloon.
_REPLACE_MIN_OVERLAP = 0.5   # intersection / area of the smaller bbox
_REPLACE_MIN_COVERAGE = 0.85  # share of the original table's words still present
_REPLACE_MIN_FILL = 0.5       # share of non-empty cells
_REPLACE_MAX_AREA_RATIO = 3.0  # replacement bbox area / original bbox area


@dataclass
class ResolvedTable:
    """A table's final rows, the strategy that produced them, and whether the
    collapse fallback was attempted for it (``attempted`` with ``strategy ==
    "lines"`` means the fallback ran and its result was rejected)."""

    rows: list[list]
    strategy: str  # "lines" | "text"
    attempted: bool = False


def _is_usable(rows: list[list]) -> bool:
    """At least 2 rows and 2 columns: what the loader will serialize."""
    return len(rows) >= 2 and max((len(row) for row in rows), default=0) >= 2


def is_collapsed_table(rows: list[list]) -> bool:
    """True for the unruled-table collapse signature (see module constants).

    A real collapse is a usable table (a header and a fused body row, 2+
    columns). A single-row or single-column "table" with many lines is a
    bordered box of prose, not a table, and must not trigger the fallback.
    """
    if not _is_usable(rows) or len(rows) > _COLLAPSE_MAX_ROWS:
        return False
    return any(
        cell is not None and len(str(cell).split("\n")) >= _COLLAPSE_MIN_CELL_LINES
        for row in rows
        for cell in row
    )


def drop_empty_columns(rows: list[list]) -> list[list]:
    """Remove columns whose every cell (header included) is empty.

    Spacer columns are common in typeset tables and only add ``|  |`` noise.
    Rows are padded to the widest row first, so ragged input stays aligned. If
    dropping would leave fewer than 2 columns the input is returned unchanged,
    so a table is never cleaned out of existence.
    """
    width = max((len(row) for row in rows), default=0)
    keep = [
        i for i in range(width)
        if any(i < len(row) and row[i] is not None and str(row[i]).strip()
               for row in rows)
    ]
    if len(keep) < 2:
        return rows
    return [[row[i] if i < len(row) else "" for i in keep] for row in rows]


def _words(rows: list[list]) -> Counter:
    return Counter(
        w for row in rows for cell in row for w in re.findall(r"\w+", str(cell or "").lower())
    )


def _word_coverage(reference: list[list], candidates: list[list[list]]) -> float:
    """Share of ``reference``'s words (with multiplicity) found in ``candidates``."""
    ref = _words(reference)
    total = sum(ref.values())
    if not total:
        return 1.0
    got: Counter = Counter()
    for rows in candidates:
        got += _words(rows)
    return sum((ref & got).values()) / total


def _fill_ratio(rows: list[list]) -> float:
    cells = [c for row in rows for c in row]
    return sum(1 for c in cells if str(c or "").strip()) / len(cells) if cells else 0.0


def _area(bbox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _overlap_fraction(a, b) -> float:
    """Intersection area over the smaller rectangle's area; bboxes are (x0,y0,x1,y1)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min(_area(a), _area(b))
    return (w * h) / smaller if smaller > 0 else 0.0


def resolve_page_tables(page: "pymupdf.Page") -> list[ResolvedTable]:
    """Every table on a page, falling back to the "text" strategy where needed.

    Tables come from the default ("lines") strategy. Any usable one matching the
    collapse signature is replaced by the overlapping tables of
    ``find_tables(strategy="text")`` -- but only if that replacement is itself
    not collapsed, has more rows, keeps at least ``_REPLACE_MIN_COVERAGE`` of the
    original's words, is at least ``_REPLACE_MIN_FILL`` filled, and is no more
    than ``_REPLACE_MAX_AREA_RATIO`` times the original's area. Otherwise the
    lines result is kept. The text strategy runs lazily, only for such a table,
    so it can never add a table to a page that had none. Empty columns are
    dropped from every result. Raises on a failing default ``find_tables``;
    per-table failures are logged and skipped.
    """
    resolved: list[ResolvedTable] = []
    text_tables = None  # type: list | None
    used: set[int] = set()
    for table in page.find_tables().tables:
        try:
            rows = table.extract()
        except Exception as exc:  # noqa: BLE001 - skip this one table
            logger.warning("table.extract() failed (page=%s): %s", page.number + 1, exc)
            continue

        attempted = False
        if is_collapsed_table(rows):
            attempted = True
            if text_tables is None:
                try:
                    text_tables = [
                        (t.bbox, t.extract()) for t in page.find_tables(strategy="text").tables
                    ]
                except Exception as exc:  # noqa: BLE001 - keep the lines result
                    logger.warning("text-strategy find_tables() failed (page=%s): %s",
                                   page.number + 1, exc)
                    text_tables = []
            replacements = [
                (i, drop_empty_columns(text_rows), bbox)
                for i, (bbox, text_rows) in enumerate(text_tables)
                if i not in used
                and _overlap_fraction(table.bbox, bbox) >= _REPLACE_MIN_OVERLAP
                and len(text_rows) > len(rows)
                and not is_collapsed_table(text_rows)
            ]
            accepted = bool(replacements) and all(
                _fill_ratio(r) >= _REPLACE_MIN_FILL
                and _area(bbox) <= _REPLACE_MAX_AREA_RATIO * _area(table.bbox)
                for _, r, bbox in replacements
            ) and _word_coverage(rows, [r for _, r, _ in replacements]) >= _REPLACE_MIN_COVERAGE
            if accepted:
                for i, new_rows, _ in replacements:
                    used.add(i)
                    resolved.append(ResolvedTable(new_rows, "text", attempted=True))
                continue
        resolved.append(ResolvedTable(drop_empty_columns(rows), "lines", attempted=attempted))
    return resolved


def extract_tables_from_page(
    page: "pymupdf.Page", *, doc_id: str, source_path: str, page_num: int
) -> list[Document]:
    """Return one ``content_type="table"`` Document per usable table on a page.

    Each table is serialized to GFM markdown (see :func:`_table_to_markdown`)
    and tagged with ``row_count`` (data rows, i.e. excluding the header),
    ``col_count`` and ``table_strategy`` ("lines", or "text" when the collapse
    fallback of :func:`resolve_page_tables` produced it). Extraction failures
    are logged and skipped, never raised, so one bad table cannot abort a
    document load.
    """
    documents: list[Document] = []
    try:
        tables = resolve_page_tables(page)
    except Exception as exc:  # noqa: BLE001 - never abort a load on table parse
        logger.warning(
            "find_tables() failed (doc_id=%s page=%s): %s",
            doc_id, page_num, exc,
        )
        return documents

    for table_index, table in enumerate(tables):
        rows = table.rows
        markdown = _table_to_markdown(rows)
        if markdown is None:
            logger.debug(
                "Skipping sub-2x2 table (doc_id=%s page=%s table=%s)",
                doc_id, page_num, table_index,
            )
            continue

        width = max((len(row) for row in rows), default=0)
        document = make_document(
            markdown,
            doc_id=doc_id,
            source_path=source_path,
            fmt="pdf",
            content_type="table",
            page=page_num,
        )
        document.metadata["row_count"] = len(rows) - 1  # data rows (no header)
        document.metadata["col_count"] = width
        document.metadata["table_strategy"] = table.strategy
        documents.append(document)

    return documents


def _save_image(image_bytes: bytes, ext: str, pdf_stem: str, page_num: int, xref: int) -> str | None:
    """Persist raw image bytes under IMAGES_DIR; return the path, or None.

    The filename is stable (``{stem}_p{page}_x{xref}.{ext}``) so re-ingesting
    the same PDF overwrites nothing and stays idempotent -- an existing file is
    left as-is. A disk error is logged and returns None so captioning still
    proceeds; the caption chunk is just created without an ``image_path``.
    """
    safe_ext = (ext or "png").lstrip(".") or "png"
    img_path = IMAGES_DIR / f"{pdf_stem}_p{page_num}_x{xref}.{safe_ext}"
    try:
        if not img_path.exists():
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(image_bytes)
        return str(img_path)
    except OSError as exc:
        logger.warning(
            "Could not write figure image %s: %s -- captioning without it.",
            img_path, exc,
        )
        return None


def load_pdf(path: str | Path) -> list[Document]:
    """Load a PDF into Documents: tables, then page text, then figure captions.

    Table Documents (``content_type="table"``) come first, then one
    ``content_type="text"`` Document per page with text, then one
    ``content_type="figure_caption"`` Document per embedded image that is large
    enough to be a real figure and that the vision model successfully captions.
    Small images (logos/icons/bullets) are skipped; a failed vision call logs a
    warning and skips that one image rather than aborting the load. Order within
    the corpus does not affect retrieval -- it is just cleaner for logging.
    """
    source_path = str(path)
    doc_id = compute_doc_id(path)
    pdf_stem = Path(path).stem
    min_dim = settings.min_figure_dimension_px

    table_docs: list[Document] = []
    text_docs: list[Document] = []
    figure_docs: list[Document] = []
    # Caption (and save) each distinct embedded image at most once per document.
    seen_xrefs: set[int] = set()

    with pymupdf.open(path) as pdf:
        for page_index in range(pdf.page_count):
            page = pdf[page_index]
            page_num = page_index + 1  # 1-indexed for metadata

            # Tables first, so they are identified before the raw-text pass.
            table_docs.extend(
                extract_tables_from_page(
                    page,
                    doc_id=doc_id,
                    source_path=source_path,
                    page_num=page_num,
                )
            )

            text = page.get_text().strip()
            if text:
                text_docs.append(
                    make_document(
                        text,
                        doc_id=doc_id,
                        source_path=source_path,
                        fmt="pdf",
                        content_type="text",
                        page=page_num,
                    )
                )

            for image_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    extracted = pdf.extract_image(xref)
                except Exception as exc:  # noqa: BLE001 - skip unreadable image
                    logger.warning(
                        "Could not extract image (doc_id=%s page=%s img=%s): %s",
                        doc_id, page_num, image_index, exc,
                    )
                    continue

                width = extracted.get("width", 0)
                height = extracted.get("height", 0)
                if width < min_dim or height < min_dim:
                    logger.debug(
                        "Skipping small image %sx%s (doc_id=%s page=%s img=%s)",
                        width, height, doc_id, page_num, image_index,
                    )
                    continue

                # Persist the raw bytes BEFORE captioning, so even a failed
                # vision call leaves the image file on disk for later display.
                image_path = _save_image(
                    extracted["image"],
                    extracted.get("ext", "png"),
                    pdf_stem,
                    page_num,
                    xref,
                )

                caption = caption_image_with_status(extracted["image"])
                if caption is None:
                    logger.warning(
                        "Captioning failed; skipping image "
                        "(doc_id=%s page=%s img=%s)",
                        doc_id, page_num, image_index,
                    )
                    continue

                figure = make_document(
                    caption.text,
                    doc_id=doc_id,
                    source_path=source_path,
                    fmt="pdf",
                    content_type="figure_caption",
                    page=page_num,
                )
                # Omitted entirely when the write failed (None), so the chunk
                # degrades to caption-only rather than carrying a dead path.
                if image_path is not None:
                    figure.metadata["image_path"] = image_path
                # Set only when the caption hit the model's length limit even
                # after a retry, so a cut-off caption is identifiable later.
                if caption.truncated:
                    figure.metadata["caption_truncated"] = True
                figure_docs.append(figure)

    return table_docs + text_docs + figure_docs
