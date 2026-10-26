"""Ingest the demo corpus into the persistent stores.

Run once (by the maintainer) before the first deploy, or as a Space startup
step. Idempotent: ``ingest_file`` skips any file whose ``doc_id`` (a hash of
the file bytes) is already present, so re-running on a Space restart re-embeds
nothing and only picks up newly added files.
"""

from __future__ import annotations

import sys
from pathlib import Path

from verirag.ingest import ingest_file
from verirag.retrieval.vector_store import list_ingested_documents

DEMO_CORPUS_DIR = Path(__file__).resolve().parent.parent / "demo_corpus"

# The corpus formats the demo accepts. README.md, LICENSES.md and .gitkeep are
# maintainer scaffolding, not corpus content, so they are skipped explicitly.
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".docx"}
_SKIP_NAMES = {"readme.md", "licenses.md", ".gitkeep"}


def _demo_files() -> list[Path]:
    if not DEMO_CORPUS_DIR.is_dir():
        return []
    return sorted(
        entry
        for entry in DEMO_CORPUS_DIR.iterdir()
        if entry.is_file()
        and entry.suffix.lower() in SUPPORTED_SUFFIXES
        and entry.name.lower() not in _SKIP_NAMES
    )


def main() -> int:
    files = _demo_files()
    if not files:
        print(
            f"[ingest_demo] {DEMO_CORPUS_DIR} is empty -- nothing to ingest. "
            "Copy the demo PDFs in and re-run.",
            file=sys.stderr,
        )
        return 0

    try:
        existing = list_ingested_documents()
    except Exception as exc:  # noqa: BLE001 - advisory; per-file dedup still holds
        print(f"[ingest_demo] warning: could not list corpus ({exc}).", file=sys.stderr)
        existing = []

    print(
        f"[ingest_demo] corpus already holds {len(existing)} document(s); "
        f"already-ingested files will be skipped."
    )

    for path in files:
        result = ingest_file(path)
        if result.skipped:
            print(f"[ingest_demo] skip    {path.name} (already ingested)")
        else:
            print(
                f"[ingest_demo] ingest  {path.name}: "
                f"{result.num_text_chunks} text + "
                f"{result.num_figure_chunks} figure chunk(s)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
