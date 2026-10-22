"""PDF loader: per-page text plus captioned embedded figures (PyMuPDF)."""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf  # the modern import name for PyMuPDF (aka ``fitz``)
from langchain_core.documents import Document

from verirag.config import settings
from verirag.loaders.base import compute_doc_id, make_document
from verirag.loaders.captioning import caption_image

logger = logging.getLogger(__name__)


def load_pdf(path: str | Path) -> list[Document]:
    """Load a PDF into Documents.

    Emits one ``content_type="text"`` Document per page (when the page has
    text), plus one ``content_type="figure_caption"`` Document per embedded
    image that is large enough to be a real figure and that the vision model
    successfully captions. Small images (logos/icons/bullets) are skipped, and
    a failed vision call logs a warning and skips that one image rather than
    aborting the load.
    """
    source_path = str(path)
    doc_id = compute_doc_id(path)
    min_dim = settings.min_figure_dimension_px

    documents: list[Document] = []
    # Caption each distinct embedded image at most once per document.
    seen_xrefs: set[int] = set()

    with pymupdf.open(path) as pdf:
        for page_index in range(pdf.page_count):
            page = pdf[page_index]
            page_num = page_index + 1  # 1-indexed for metadata

            text = page.get_text().strip()
            if text:
                documents.append(
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

                caption = caption_image(extracted["image"])
                if caption is None:
                    logger.warning(
                        "Captioning failed; skipping image "
                        "(doc_id=%s page=%s img=%s)",
                        doc_id, page_num, image_index,
                    )
                    continue

                documents.append(
                    make_document(
                        caption,
                        doc_id=doc_id,
                        source_path=source_path,
                        fmt="pdf",
                        content_type="figure_caption",
                        page=page_num,
                    )
                )

    return documents
