"""Phase 14 Step 0 diagnostic (THROWAWAY): validate find_tables() on the corpus.

Opens every PDF in the real corpus directory, runs page.find_tables() on every
page, and prints per-table shape plus a summary. This is a go/no-go gate: if
find_tables() is unavailable or finds zero tables across the whole corpus, the
Phase 14 table work should not proceed.

NOTE ON PATHS: the Phase 14 brief referenced data/corpus/, which does not exist
in this repo. The actual ingested corpus lives in data/raw/uploads/, so that is
what this scans.

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/verirag/bin/python \
        scripts/test_phase14_table_diagnostic.py
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "raw" / "uploads"


def _first_row(rows: list) -> list:
    for row in rows:
        if row:
            return row
    return []


def main() -> int:
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found under {CORPUS_DIR}")
        return 1

    print(f"pymupdf version: {pymupdf.__version__}")
    print(f"Scanning {len(pdfs)} PDF(s) under {CORPUS_DIR}\n")

    total_pages = 0
    total_tables = 0
    pdfs_with_tables = 0

    for pdf_path in pdfs:
        file_had_table = False
        with pymupdf.open(pdf_path) as doc:
            for page_index in range(doc.page_count):
                page = doc[page_index]
                page_num = page_index + 1
                total_pages += 1

                # find_tables() is the method under test; let an AttributeError
                # propagate so the go/no-go gate sees it.
                finder = page.find_tables()
                tables = list(finder.tables)
                if not tables:
                    continue

                file_had_table = True
                total_tables += len(tables)
                print(
                    f"{pdf_path.name}  page {page_num}: "
                    f"{len(tables)} table(s)"
                )
                for t_index, table in enumerate(tables):
                    try:
                        rows = table.extract()
                    except Exception as exc:  # noqa: BLE001 - diagnostic only
                        print(f"    table {t_index}: extract() failed: {exc}")
                        continue
                    row_count = len(rows)
                    col_count = len(rows[0]) if rows else 0
                    first = _first_row(rows)
                    print(
                        f"    table {t_index}: rows={row_count} "
                        f"cols={col_count} first_row={first}"
                    )
        if file_had_table:
            pdfs_with_tables += 1

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  total pages scanned      : {total_pages}")
    print(f"  total tables found       : {total_tables}")
    print(f"  PDFs with >=1 table      : {pdfs_with_tables} of {len(pdfs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ============================================================================
# SUMMARY OUTPUT (pasted after a successful run per Step 0 instructions):
# ----------------------------------------------------------------------------
#   pymupdf version          : 2.0  (Page.find_tables present, no AttributeError)
#   total pages scanned      : 95
#   total tables found       : 47
#   PDFs with >=1 table      : 6 of 6
#
# Per-PDF highlights:
#   1706.03762v7.pdf (Attention Is All You Need): tables on pp. 6, 8, 9, 10
#   2006.03654v6.pdf + 2006.03654v6-3d3d9c91.pdf (DeBERTa, duplicate copies):
#       tables on pp. 6-9, 15-19, 22, 23 (many results tables)
#   2311.14966v1.pdf: tables on pp. 3, 4, 5, 13
#   Anderson-and-Krathwohl_Revised-Blooms-Taxonomy.pdf: tables on pp. 2, 3
#   Automated Educational Question Generation ... Bloom's Skill Levels.pdf:
#       tables on pp. 3 ("Bloom's level / Description", 10 rows), 6, 8, 9, 12
#
# Observations that shaped the loader implementation:
#   - Extracted cell strings frequently contain embedded "\n" (e.g.
#     'Sequential\nOperations') -> must collapse newlines/whitespace per cell
#     so a GFM row is not broken; pipe chars must be escaped.
#   - None cells appear (merged/empty) -> replaced with "".
#   - Many tiny 1-row or narrow blocks appear -> the >=2 rows AND >=2 cols
#     filter in the loader drops false positives.
#
# VERDICT: find_tables() works and the corpus is table-rich. Proceeded to
# Steps 1-4.
# ============================================================================
