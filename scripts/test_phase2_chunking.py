"""Phase 2 smoke test: chunk -> contextualize -> embed, over the fixtures.

Usage (from the project root, in the `verirag` env):

    python scripts/test_phase2_chunking.py

Importing the pipeline imports verirag.config, so DEEPSEEK_API_KEY must be set
(a placeholder is fine). Without a real key the contextualization step will
hit the graceful-fallback path (context_prefix=None), which is expected here.
The embedding step downloads BAAI/bge-base-en-v1.5 on first run.
"""

from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from pathlib import Path

from verirag.chunking.chunker import chunk_documents, count_tokens
from verirag.chunking.contextualize import add_context_to_chunks
from verirag.embeddings.embedder import embed_passages
from verirag.loaders import load_document

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
FIXTURE_FILES = [
    "sample.txt",
    "sample.md",
    "sample.docx",
    "sample.pptx",
    "sample.pdf",
]


def _doc_text_by_id(documents) -> dict[str, str]:
    """Concatenated source text per doc_id (text Documents only)."""
    collected: dict[str, list[str]] = defaultdict(list)
    for doc in documents:
        if doc.metadata["content_type"] == "text":
            collected[doc.metadata["doc_id"]].append(doc.page_content)
    return {doc_id: "\n\n".join(parts) for doc_id, parts in collected.items()}


def _report_file(name: str) -> None:
    path = FIXTURES_DIR / name
    print("=" * 72)
    print(f"FILE: {name}")
    if not path.exists():
        print("  MISSING - run: python tests/fixtures/generate_fixtures.py")
        return

    documents = load_document(path)
    doc_text_by_id = _doc_text_by_id(documents)

    # Chunk (chunk_documents groups by doc_id and passes captions through).
    chunks = chunk_documents(documents)

    # Contextualize per source document (same-doc chunks together).
    by_doc: "OrderedDict[str, list]" = OrderedDict()
    for chunk in chunks:
        by_doc.setdefault(chunk.metadata["doc_id"], []).append(chunk)
    chunks = []
    for doc_id, doc_chunks in by_doc.items():
        chunks.extend(
            add_context_to_chunks(doc_chunks, doc_text_by_id.get(doc_id, ""))
        )

    # Embed the page_content (prefix + raw_text) of every chunk.
    embeddings = embed_passages([c.page_content for c in chunks])

    # Token stats are over raw_text (what the chunker's min/max bounds govern).
    token_counts = [count_tokens(c.metadata["raw_text"]) for c in chunks]
    n = len(chunks)
    print(f"  chunks produced: {n}")
    if token_counts:
        print(
            f"  raw_text tokens: min={min(token_counts)} "
            f"max={max(token_counts)} "
            f"avg={sum(token_counts) / len(token_counts):.1f}"
        )
    print(f"  embeddings array: shape={embeddings.shape} dtype={embeddings.dtype}")

    for i, chunk in enumerate(chunks[:2]):
        raw = chunk.metadata["raw_text"][:150].replace("\n", " ")
        prefix = chunk.metadata.get("context_prefix")
        vec = embeddings[i]
        print(f"  --- chunk[{i}] (chunk_id={chunk.metadata['chunk_id']}, "
              f"content_type={chunk.metadata['content_type']}) ---")
        print(f"      raw_text[:150]: {raw!r}")
        print(f"      context_prefix: {prefix if prefix is not None else 'None'}")
        print(f"      embedding: shape={vec.shape} dtype={vec.dtype}")


def main() -> None:
    print(f"Fixtures: {FIXTURES_DIR}")
    for name in FIXTURE_FILES:
        _report_file(name)
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
