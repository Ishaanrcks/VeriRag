"""Phase 1 smoke test: run every loader against the synthetic fixtures.

Usage (from the project root, in the `verirag` env):

    python scripts/test_phase1_loaders.py

Requires DEEPSEEK_API_KEY in the environment / .env (importing the loaders
imports verirag.config, and the PDF fixture's figure triggers a vision call).
If the fixtures are missing, run tests/fixtures/generate_fixtures.py first.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

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


def _report_file(name: str) -> None:
    path = FIXTURES_DIR / name
    print("=" * 72)
    print(f"FILE: {name}")
    if not path.exists():
        print("  MISSING - run: python tests/fixtures/generate_fixtures.py")
        return

    docs = load_document(path)
    print(f"  Documents returned: {len(docs)}")

    breakdown = Counter(d.metadata["content_type"] for d in docs)
    print(f"  content_type breakdown: {dict(breakdown)}")
    if docs:
        print(f"  doc_id: {docs[0].metadata['doc_id']}")

    # First 200 chars of every text Document.
    for i, d in enumerate(docs):
        if d.metadata["content_type"] != "text":
            continue
        snippet = d.page_content[:200].replace("\n", " ")
        print(f"  [{i}] text (page={d.metadata['page']}): {snippet!r}")

    # Full text of every figure_caption Document.
    for i, d in enumerate(docs):
        if d.metadata["content_type"] != "figure_caption":
            continue
        print(f"  [{i}] FIGURE CAPTION (page={d.metadata['page']}) - full text:")
        print("  " + "-" * 68)
        for line in d.page_content.splitlines() or [d.page_content]:
            print(f"  | {line}")
        print("  " + "-" * 68)


def main() -> None:
    print(f"Loading fixtures from: {FIXTURES_DIR}")
    for name in FIXTURE_FILES:
        _report_file(name)
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
