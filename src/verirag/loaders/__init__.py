"""Multi-format document loaders for VeriRAG.

Public API:
- ``load_document(path)``: dispatch by extension to the right loader.
- the per-format loaders (``load_pdf``, ``load_docx``, ``load_pptx``,
  ``load_txt``, ``load_md``).

All loaders return ``list[langchain_core.documents.Document]`` using the shared
metadata schema defined in :mod:`verirag.loaders.base`.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from verirag.loaders.base import EXTENSION_TO_FORMAT
from verirag.loaders.office import load_docx, load_pptx
from verirag.loaders.pdf import load_pdf
from verirag.loaders.plaintext import load_md, load_txt

# Dispatch table keyed by canonical format label.
_LOADERS = {
    "pdf": load_pdf,
    "docx": load_docx,
    "pptx": load_pptx,
    "txt": load_txt,
    "md": load_md,
}


def load_document(path: str | Path) -> list[Document]:
    """Load any supported document, dispatching on its file extension.

    Raises ``ValueError`` for unsupported extensions.
    """
    ext = Path(path).suffix.lower()
    fmt = EXTENSION_TO_FORMAT.get(ext)
    if fmt is None:
        supported = ", ".join(sorted(EXTENSION_TO_FORMAT)) or "(none)"
        raise ValueError(
            f"Unsupported file extension {ext!r} for {path!r}. "
            f"Supported extensions: {supported}."
        )
    return _LOADERS[fmt](path)


__all__ = [
    "load_document",
    "load_pdf",
    "load_docx",
    "load_pptx",
    "load_txt",
    "load_md",
]
