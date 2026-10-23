"""Shared helpers for building the unified document representation.

Every loader emits ``langchain_core.documents.Document`` objects whose
``metadata`` follows a fixed schema so downstream phases (chunking, embedding,
retrieval, the critic) can rely on it:

- ``doc_id``       : stable id for the source file (filename stem + content hash)
- ``source_path``  : original file path
- ``format``       : one of "pdf" | "docx" | "pptx" | "txt" | "md"
- ``content_type`` : "text" | "figure_caption"
- ``page``         : page/slide number (1-indexed) where applicable, else None
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document

# "external_web" is not a local file type -- it tags content fetched from the
# web by external/fetcher.py, which has no file extension to map from.
FormatLiteral = Literal["pdf", "docx", "pptx", "txt", "md", "external_web"]
ContentType = Literal["text", "figure_caption"]

# Maps a (lower-cased) file extension to its canonical format label.
EXTENSION_TO_FORMAT: dict[str, FormatLiteral] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".txt": "txt",
    ".md": "md",
}


def compute_duplicate_group_id(leading_text: str) -> str:
    """Group id shared by documents that appear to be the same work.

    Derived from a normalized prefix of the document's own opening text (title
    and byline), reduced to lowercase alphanumerics so that layout differences
    collapse: the same paper's camera-ready and arXiv versions break the title
    across lines differently and hash to different doc_ids, but normalize to
    the same prefix here.

    Using the normalized title AS the group id means two copies share a group
    automatically, with no need to go back and rewrite the metadata of a
    document that was ingested earlier.
    """
    normalized = re.sub(r"[^a-z0-9]", "", (leading_text or "").lower())
    if not normalized:
        return ""
    return hashlib.sha256(normalized[:80].encode("utf-8")).hexdigest()[:12]


def compute_doc_id(path: str | Path) -> str:
    """Return a stable id for a source file: ``<stem>-<sha256[:12]>``.

    The hash is over the raw file bytes, so the same file always yields the
    same id and edited files get a new one.
    """
    p = Path(path)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    return f"{p.stem}-{digest}"


def make_document(
    page_content: str,
    *,
    doc_id: str,
    source_path: str,
    fmt: FormatLiteral,
    content_type: ContentType,
    page: int | None = None,
) -> Document:
    """Build a Document with the VeriRAG metadata schema populated."""
    return Document(
        page_content=page_content,
        metadata={
            "doc_id": doc_id,
            "source_path": source_path,
            "format": fmt,
            "content_type": content_type,
            "page": page,
        },
    )
