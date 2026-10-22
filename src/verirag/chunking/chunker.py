"""Token-budgeted recursive chunking of loaded Documents.

``chunk_documents`` splits only ``content_type == "text"`` Documents using a
recursive-structural splitter (paragraph -> line -> sentence -> word -> hard
cut) with a tiktoken-based token length function, then runs a merge pass to
fold undersized chunks into a neighbour. ``figure_caption`` Documents are
passed through untouched as a single chunk each (they are already short).

Token counting uses tiktoken's ``cl100k_base`` purely as a cheap, consistent
proxy for chunk size -- it need not match DeepSeek's or BGE's tokenizer.
"""

from __future__ import annotations

from collections import OrderedDict

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from verirag.config import settings

# Structural separators, tried in order; the trailing "" is the hard
# character-level cut used only when nothing else keeps a chunk under budget.
_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", " ", ""]

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate token count via tiktoken cl100k_base."""
    return len(_ENCODING.encode(text))


_SPLITTER = RecursiveCharacterTextSplitter(
    separators=_SEPARATORS,
    chunk_size=settings.chunk_max_tokens,
    chunk_overlap=settings.chunk_overlap_tokens,
    length_function=count_tokens,
    keep_separator=True,
    is_separator_regex=False,
)


def _merge_undersized(texts: list[str]) -> list[str]:
    """Fold any below-min chunk into a neighbour.

    A chunk under ``chunk_min_tokens`` is merged into the *next* chunk, or into
    the *previous* one if it is the last chunk. Repeats until every chunk meets
    the minimum or only a single chunk remains for this source Document.
    """
    merged = list(texts)
    while len(merged) > 1:
        for i, text in enumerate(merged):
            if count_tokens(text) >= settings.chunk_min_tokens:
                continue
            if i < len(merged) - 1:  # merge into the next chunk
                merged[i + 1] = f"{text}\n\n{merged[i + 1]}"
            else:  # last chunk: merge into the previous one
                merged[i - 1] = f"{merged[i - 1]}\n\n{text}"
            del merged[i]
            break  # restart scan after a structural change
        else:
            break  # no undersized chunk found this pass
    return merged


def _make_chunk(parent: Document, text: str, doc_id: str, index: int) -> Document:
    """Build a chunk Document, copying parent metadata and adding chunk fields."""
    metadata = dict(parent.metadata)  # doc_id, source_path, format, content_type, page
    metadata["chunk_id"] = f"{doc_id}-{index}"
    metadata["chunk_index"] = index
    metadata["raw_text"] = text  # original text BEFORE any contextual prefix
    return Document(page_content=text, metadata=metadata)


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Chunk a list of loaded Documents into token-budgeted chunk Documents.

    ``chunk_index`` is a running 0-based counter *per source file* (per
    ``doc_id``), so ``chunk_id = f"{doc_id}-{chunk_index}"`` is unique even
    when a file produced several text Documents (e.g. PDF pages / pptx slides).
    """
    # Group by doc_id while preserving first-seen order within each file.
    groups: "OrderedDict[str, list[Document]]" = OrderedDict()
    for doc in documents:
        groups.setdefault(doc.metadata["doc_id"], []).append(doc)

    chunks: list[Document] = []
    for doc_id, parents in groups.items():
        index = 0
        for parent in parents:
            if parent.metadata.get("content_type") == "figure_caption":
                # Already short; keep as a single chunk, do not re-split.
                chunks.append(_make_chunk(parent, parent.page_content, doc_id, index))
                index += 1
                continue

            pieces = _SPLITTER.split_text(parent.page_content)
            pieces = _merge_undersized(pieces)
            for piece in pieces:
                chunks.append(_make_chunk(parent, piece, doc_id, index))
                index += 1

    return chunks
