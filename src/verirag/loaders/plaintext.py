"""Plain-text and Markdown loaders."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from verirag.loaders.base import FormatLiteral, compute_doc_id, make_document


def _load_raw_text(path: str | Path, fmt: FormatLiteral) -> list[Document]:
    source_path = str(path)
    doc_id = compute_doc_id(path)
    text = Path(path).read_text(encoding="utf-8")
    return [
        make_document(
            text,
            doc_id=doc_id,
            source_path=source_path,
            fmt=fmt,
            content_type="text",
            page=None,
        )
    ]


def load_txt(path: str | Path) -> list[Document]:
    """Load a .txt file as a single text Document."""
    return _load_raw_text(path, "txt")


def load_md(path: str | Path) -> list[Document]:
    """Load a .md file as a single text Document (raw markdown, unparsed)."""
    return _load_raw_text(path, "md")
