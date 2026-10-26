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
from pathlib import Path

import pymupdf  # the modern import name for PyMuPDF (aka ``fitz``)
from langchain_core.documents import Document

from verirag.config import settings
from verirag.loaders.base import compute_doc_id, make_document
from verirag.loaders.captioning import caption_image

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


def extract_tables_from_page(
    page: "pymupdf.Page", *, doc_id: str, source_path: str, page_num: int
) -> list[Document]:
    """Return one ``content_type="table"`` Document per usable table on a page.

    Each table is serialized to GFM markdown (see :func:`_table_to_markdown`)
    and tagged with ``row_count`` (data rows, i.e. excluding the header) and
    ``col_count``. Extraction failures are logged and skipped, never raised, so
    one bad table cannot abort a document load.
    """
    documents: list[Document] = []
    try:
        tables = list(page.find_tables().tables)
    except Exception as exc:  # noqa: BLE001 - never abort a load on table parse
        logger.warning(
            "find_tables() failed (doc_id=%s page=%s): %s",
            doc_id, page_num, exc,
        )
        return documents

    for table_index, table in enumerate(tables):
        try:
            rows = table.extract()
        except Exception as exc:  # noqa: BLE001 - skip this one table
            logger.warning(
                "table.extract() failed (doc_id=%s page=%s table=%s): %s",
                doc_id, page_num, table_index, exc,
            )
            continue

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

                caption = caption_image(extracted["image"])
                if caption is None:
                    logger.warning(
                        "Captioning failed; skipping image "
                        "(doc_id=%s page=%s img=%s)",
                        doc_id, page_num, image_index,
                    )
                    continue

                figure = make_document(
                    caption,
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
                figure_docs.append(figure)

    return table_docs + text_docs + figure_docs
