"""Reset the stores and re-ingest the full corpus (data/raw/uploads/).

Phase 14 requires a full re-ingest so every PDF is re-processed with the new
table extraction and image persistence. This wipes the dense (Chroma) and
sparse (BM25) stores first so no pre-Phase-14 chunks linger, then ingests every
PDF under data/raw/uploads/.

"Wipe" is done by MOVING the old stores aside to a timestamped .bak directory
(reversible) rather than deleting them outright.

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/verirag/bin/python \
        scripts/reingest_corpus.py
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CORPUS_DIR = PROJECT_ROOT / "data" / "raw" / "uploads"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
BM25_PATH = PROJECT_ROOT / "data" / "processed" / "bm25_corpus.jsonl"


def _backup_stores() -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if CHROMA_DIR.exists():
        dest = CHROMA_DIR.with_name(f"chroma.bak-{stamp}")
        shutil.move(str(CHROMA_DIR), str(dest))
        print(f"[reset] moved {CHROMA_DIR.name} -> {dest.name}")
    if BM25_PATH.exists():
        dest = BM25_PATH.with_name(f"bm25_corpus.bak-{stamp}.jsonl")
        shutil.move(str(BM25_PATH), str(dest))
        print(f"[reset] moved {BM25_PATH.name} -> {dest.name}")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Quiet the noisiest third parties so the ingest log stays readable.
    for noisy in ("httpx", "urllib3", "sentence_transformers", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"[reingest] no PDFs under {CORPUS_DIR}")
        return 1

    print(f"[reingest] wiping stores, then ingesting {len(pdfs)} PDF(s)")
    _backup_stores()

    # Import AFTER the move so no Chroma client is holding the old directory.
    from verirag.ingest import ingest_file
    from verirag.retrieval.vector_store import get_collection

    totals = {"text": 0, "figure": 0}
    for path in pdfs:
        print(f"\n[reingest] ingesting {path.name} ...")
        result = ingest_file(path, force=True)
        totals["text"] += result.num_text_chunks
        totals["figure"] += result.num_figure_chunks
        print(
            f"[reingest] {path.name}: "
            f"{result.num_text_chunks} text + "
            f"{result.num_figure_chunks} figure chunk(s)"
            + ("  [SKIPPED]" if result.skipped else "")
        )

    # Verify what actually landed in the store, by content_type.
    collection = get_collection()
    stored = collection.get(include=["metadatas"])
    metas = stored.get("metadatas") or []
    by_type: dict[str, int] = {}
    figs_with_image = 0
    for meta in metas:
        meta = meta or {}
        ct = str(meta.get("content_type") or "?")
        by_type[ct] = by_type.get(ct, 0) + 1
        if ct == "figure_caption" and meta.get("image_path"):
            figs_with_image += 1

    print("\n" + "=" * 64)
    print("REINGEST SUMMARY")
    print("=" * 64)
    print(f"  text/figure created (IngestResult): {totals}")
    print(f"  chunks in store by content_type   : {by_type}")
    print(f"  total chunks in store             : {len(metas)}")
    print(f"  figure_caption chunks w/ image_path: {figs_with_image}")
    if by_type.get("table", 0) == 0:
        print("  WARNING: zero table chunks in store -- STOP point hit.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
