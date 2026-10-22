"""DOCX and PPTX loaders (python-docx / python-pptx)."""

from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument
from langchain_core.documents import Document
from pptx import Presentation

from verirag.loaders.base import compute_doc_id, make_document


def load_docx(path: str | Path) -> list[Document]:
    """Load a .docx as a single text Document (all paragraphs joined).

    NOTE: image/figure handling is intentionally out of scope for docx in this
    MVP — only paragraph text is extracted. (PDF is the format with figure
    captioning in Phase 1.)
    """
    source_path = str(path)
    doc_id = compute_doc_id(path)

    docx = DocxDocument(str(path))
    paragraphs = [p.text for p in docx.paragraphs if p.text and p.text.strip()]
    text = "\n".join(paragraphs)

    return [
        make_document(
            text,
            doc_id=doc_id,
            source_path=source_path,
            fmt="docx",
            content_type="text",
            page=None,
        )
    ]


def load_pptx(path: str | Path) -> list[Document]:
    """Load a .pptx as one text Document per slide (page = slide number).

    NOTE: image/figure handling is intentionally out of scope for pptx in this
    MVP — only shape text is extracted.
    """
    source_path = str(path)
    doc_id = compute_doc_id(path)

    presentation = Presentation(str(path))
    documents: list[Document] = []

    for slide_index, slide in enumerate(presentation.slides):
        slide_num = slide_index + 1  # 1-indexed
        lines: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in paragraph.runs)
                    if line.strip():
                        lines.append(line)
        text = "\n".join(lines)
        if not text.strip():
            continue
        documents.append(
            make_document(
                text,
                doc_id=doc_id,
                source_path=source_path,
                fmt="pptx",
                content_type="text",
                page=slide_num,
            )
        )

    return documents
