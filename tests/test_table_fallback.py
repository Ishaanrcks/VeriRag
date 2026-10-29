"""Table extraction: the collapse signature and the "text" strategy fallback.

Most tests drive ``resolve_page_tables`` with a fake page built from rows and
bboxes recorded from real PDFs (``fixtures/table_fallback_cases.json``), so
they need no PDF and no model. A couple of tests build tiny PDFs with PyMuPDF.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from verirag.loaders import pdf as P

CASES = json.loads((Path(__file__).parent / "fixtures" / "table_fallback_cases.json").read_text())


class _Table:
    def __init__(self, spec):
        self.bbox = tuple(spec["bbox"])
        self._rows = spec["rows"]

    def extract(self):
        return self._rows


class _Found:
    def __init__(self, specs):
        self.tables = [_Table(s) for s in specs]


class FakePage:
    number = 0

    def __init__(self, case, forbid_text=False):
        self._case, self._forbid_text = case, forbid_text
        self.text_calls = 0

    def find_tables(self, strategy=None):
        if strategy == "text":
            self.text_calls += 1
            assert not self._forbid_text, "text strategy must not run here"
            return _Found(self._case["text"])
        return _Found(self._case["lines"])


def _cell(c):
    return " ".join(str(c or "").split())


# --------------------------------------------------------------------------
# the collapse signature
# --------------------------------------------------------------------------
def test_collapse_signature_on_real_cases():
    itc = CASES["itc_p5_attendance_unruled"]
    assert P.is_collapsed_table(itc["lines"][0]["rows"])  # 2 rows, a 28-line cell
    assert not P.is_collapsed_table(itc["text"][0]["rows"])  # 15 rows
    assert not P.is_collapsed_table(CASES["pew_p15_ruled_usable"]["lines"][0]["rows"])


def test_boxed_prose_is_not_a_collapsed_table():
    # A 1x3 bordered box of prose has a many-line cell but is not a table; the
    # first draft of the fallback turned one into a 59x8 table of fragments.
    rows = CASES["boxed_prose_f8d3_p13"]["lines"][0]["rows"]
    assert len(rows) == 1 and any(len(str(c).split("\n")) >= 6 for c in rows[0])
    assert not P.is_collapsed_table(rows)


# --------------------------------------------------------------------------
# empty columns
# --------------------------------------------------------------------------
def test_drop_empty_columns_removes_spacers():
    rows = CASES["pew_p15_ruled_usable"]["lines"][0]["rows"]
    assert max(len(r) for r in rows) == 6
    assert max(len(r) for r in P.drop_empty_columns(rows)) == 5


def test_drop_empty_columns_never_drops_below_two_columns():
    # NIKE p11: a spacer column plus one content column. Dropping the spacer
    # would leave one column, which the loader then discards as "not a table".
    rows = CASES["nike_p11_single_content_column"]["lines"][0]["rows"]
    assert P.drop_empty_columns(rows) == rows
    assert P._table_to_markdown(P.drop_empty_columns(rows)) is not None


# --------------------------------------------------------------------------
# resolve_page_tables on real cases
# --------------------------------------------------------------------------
def test_unruled_attendance_table_is_recovered_and_paired():
    resolved = P.resolve_page_tables(FakePage(CASES["itc_p5_attendance_unruled"]))
    assert [r.strategy for r in resolved] == ["text"]
    rows = resolved[0].rows
    assert (len(rows), max(len(r) for r in rows)) == (15, 3)  # header + 14 directors
    by_name = {_cell(r[0]).split("*")[0].split(" .")[0].strip(): _cell(r[1]) for r in rows[1:]}
    assert by_name["S H Khan"] == "1"  # the lowest attendance, correctly paired
    assert by_name["Y C Deveshwar"] == "5" and by_name["C R Green"] == "3"


def test_second_new_document_case_is_recovered():
    # afe620b9 p9: a directors table with no row rules -> header + fused row.
    case = CASES["afe620b9_p9_directors_unruled"]
    resolved = P.resolve_page_tables(FakePage(case))
    assert all(r.strategy == "text" and r.attempted for r in resolved)
    big = max(resolved, key=lambda r: len(r.rows))
    assert len(big.rows) >= 8
    shah = next(r for r in big.rows if "RA Shah" in _cell(r[0]))
    assert _cell(shah[2]) == "7" and _cell(shah[3]) == "Yes"


def test_ruled_table_is_left_alone():
    page = FakePage(CASES["pew_p15_ruled_usable"], forbid_text=True)
    resolved = P.resolve_page_tables(page)
    assert [(r.strategy, r.attempted) for r in resolved] == [("lines", False)]
    assert page.text_calls == 0


def test_boxed_prose_never_reaches_the_text_strategy():
    page = FakePage(CASES["boxed_prose_f8d3_p13"], forbid_text=True)
    resolved = P.resolve_page_tables(page)
    assert [(r.strategy, r.attempted) for r in resolved] == [("lines", False)]


def test_shredded_replacement_is_rejected():
    # afe620b9 p10: the text strategy turned a legible 4x4 table into a grid of
    # prose fragments (word coverage 0.40, a third of cells filled). Kept as lines.
    case = CASES["afe620b9_p10_shredded_prose"]
    lines_rows = case["lines"][0]["rows"]
    shredded = P.drop_empty_columns(case["text"][0]["rows"])
    assert P._word_coverage(lines_rows, [shredded]) < P._REPLACE_MIN_COVERAGE
    assert P._fill_ratio(shredded) < P._REPLACE_MIN_FILL
    resolved = P.resolve_page_tables(FakePage(case))
    assert [(r.strategy, r.attempted) for r in resolved] == [("lines", True)]


def test_text_result_that_is_also_collapsed_is_not_used():
    case = CASES["afe620b9_p13_text_also_collapsed"]
    resolved = P.resolve_page_tables(FakePage(case))
    assert [(r.strategy, r.attempted) for r in resolved] == [("lines", True)]


def test_fallback_never_runs_on_a_page_without_tables():
    page = FakePage({"lines": [], "text": [{"bbox": [0, 0, 100, 100], "rows": [["a", "b"]] * 5}]},
                    forbid_text=True)
    assert P.resolve_page_tables(page) == []


# --------------------------------------------------------------------------
# end to end on small generated PDFs
# --------------------------------------------------------------------------
def _unruled_table_pdf(path):
    """Outer box + column rules + a header rule, but NO row rules: the shape
    that makes the "lines" strategy return a header and one fused body row."""
    doc = pymupdf.open()
    page = doc.new_page()
    x = [60, 260, 400, 540]
    top = 100
    for i, h in enumerate(["Director", "Meetings attended", "At last AGM"]):
        page.insert_text((x[i] + 4, top + 14), h, fontsize=10)
    rows = [("Alice Archer", "5", "Yes"), ("Bob Brandt", "3", "No"), ("Carol Cho", "1", "NA"),
            ("Dan Duval", "5", "Yes"), ("Eve Eskin", "5", "Yes"), ("Fay Flint", "4", "Yes"),
            ("Gus Gil", "2", "No"), ("Hal Hix", "5", "Yes")]
    for r, row in enumerate(rows):
        for i, c in enumerate(row):
            page.insert_text((x[i] + 4, top + 40 + r * 16), c, fontsize=10)
    bottom = top + 40 + len(rows) * 16
    shape = page.new_shape()
    shape.draw_rect(pymupdf.Rect(x[0], top, x[-1], bottom))
    for xv in x[1:-1]:
        shape.draw_line((xv, top), (xv, bottom))
    shape.draw_line((x[0], top + 22), (x[-1], top + 22))
    shape.finish(width=0.8)
    shape.commit()
    doc.save(str(path))
    return rows


def test_generated_unruled_pdf_collapses_under_lines_and_is_fixed(tmp_path):
    expected = _unruled_table_pdf(tmp_path / "t.pdf")
    page = pymupdf.open(tmp_path / "t.pdf")[0]

    lines_only = [t.extract() for t in page.find_tables().tables]
    assert len(lines_only) == 1 and P.is_collapsed_table(lines_only[0])

    docs = P.extract_tables_from_page(page, doc_id="d", source_path="t.pdf", page_num=1)
    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta["table_strategy"] == "text" and meta["col_count"] == 3
    assert meta["content_type"] == "table" and meta["page"] == 1
    body = docs[0].page_content
    for name, count, attended in expected:
        compact = name.replace(" ", "")
        line = next(ln for ln in body.splitlines() if compact in ln.replace(" ", ""))
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert cells[1:] == [count, attended], line


def test_prose_only_pdf_yields_no_tables(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 500, 700), "Plain prose. " * 200, fontsize=11)
    doc.save(str(tmp_path / "p.pdf"))
    page = pymupdf.open(tmp_path / "p.pdf")[0]
    assert P.extract_tables_from_page(page, doc_id="d", source_path="p.pdf", page_num=1) == []
