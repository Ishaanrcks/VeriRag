"""Vector-drawn charts must reach the figure-caption flow.

``load_pdf`` used to caption only embedded raster images, so a chart drawn as PDF
path operators (bars, pie slices, lines) was silently dropped and only its text labels
survived. Regression cases: Pew p16 (a bar chart of filled paths, no embedded image at
all) and ITC annual report p9 (a pie and a bar chart; the document has zero raster
images). The two PDFs are third-party publications and are not in the repo, so the
fixture holds each page's drawing GEOMETRY and chart labels, from which the page is
rebuilt; the same assertions also run on the real PDFs when they are available locally.

The vision call is stubbed everywhere: no network, no model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pymupdf
import pytest

import verirag.loaders.pdf as pdf_loader
from verirag.loaders import vector_charts as vc
from verirag.loaders.captioning import Caption

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "vector_chart_pages.json").read_text())
REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# building pages
# --------------------------------------------------------------------------
def rebuild_page(doc: "pymupdf.Document", spec: dict) -> "pymupdf.Page":
    """A page with the recorded drawing paths and chart labels (no images)."""
    page = doc.new_page(width=spec["page_size"][0], height=spec["page_size"][1])
    for d in spec["paths"]:
        shape = page.new_shape()
        for item in d["items"]:
            kind, v = item[0], item[1:]
            if kind == "l":
                shape.draw_line((v[0], v[1]), (v[2], v[3]))
            elif kind == "c":
                shape.draw_bezier((v[0], v[1]), (v[2], v[3]), (v[4], v[5]), (v[6], v[7]))
            elif kind == "re":
                shape.draw_rect(pymupdf.Rect(*v))
            elif kind == "qu":
                shape.draw_quad(pymupdf.Quad((v[0], v[1]), (v[2], v[3]), (v[6], v[7]), (v[4], v[5])))
        shape.finish(fill=d["fill"], color=d["color"], width=d["width"] or 0.0,
                     closePath=d["close"], fill_opacity=d["fill_opacity"])
        shape.commit()
    for span in spec["spans"]:
        page.insert_text((span["x"], span["y"]), span["text"], fontsize=span["size"])
    return page


def fixture_pdf(tmp_path: Path, name: str) -> Path:
    doc = pymupdf.open()
    rebuild_page(doc, FIXTURE[name])
    path = tmp_path / f"{name}.pdf"
    doc.save(str(path))
    return path


def draw_bar_chart(page, x=60, y=100, bars=5, labels=True, scale=1.0):
    """A multi-colour bar chart with numeric labels, as vector rectangles (a chart the
    strict rule accepts: 5 shapes in 3+ colours, 3+ numeric labels)."""
    colours = [(0.1, 0.4, 0.7), (0.8, 0.2, 0.2), (0.2, 0.6, 0.3), (0.9, 0.6, 0.1), (0.5, 0.3, 0.6)]
    # a baseline joins the bars into one drawing cluster, as a real chart's axis does
    # (PyMuPDF clusters only shapes that touch)
    axis = page.new_shape()
    axis.draw_line((x - 4 * scale, y + 120 * scale), (x + 40 * bars * scale, y + 120 * scale))
    axis.finish(color=(0.3, 0.3, 0.3), width=0.8)
    axis.commit()
    for i in range(bars):
        shape = page.new_shape()
        height = (30 + 14 * i) * scale
        left = x + 40 * i * scale
        shape.draw_rect(pymupdf.Rect(left, y + 120 * scale - height, left + 28 * scale, y + 120 * scale))
        shape.finish(fill=colours[i % len(colours)], color=None)
        shape.commit()
        if labels:
            page.insert_text((left, y + 112 * scale - height), str(10 + 7 * i), fontsize=9 * scale)
    if labels:
        page.insert_text((x, y + 138 * scale), "Group A   Group B   Group C   Group D   Group E", fontsize=8 * scale)


def vector_figures(tmp_path, draw):
    return [d for d in pdf_loader.load_pdf(one_page_pdf(tmp_path, draw))
            if d.metadata["content_type"] == "figure_caption"
            and d.metadata.get("figure_source") == "vector_render"]


def assert_spoiler_blocks(tmp_path, spoiled, clean):
    """CONTROL: the clean page is detected, so only the spoiler can explain the miss."""
    assert len(vector_figures(tmp_path / "clean", clean)) == 1, "control chart should qualify"
    assert vector_figures(tmp_path / "spoiled", spoiled) == []


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """Stub the vision call; record every image the pipeline would caption."""
    sent: list[bytes] = []

    def stub(image_bytes):
        sent.append(image_bytes)
        return Caption("[caption]")

    monkeypatch.setattr(pdf_loader, "caption_image_with_status", stub)
    monkeypatch.setattr(pdf_loader, "IMAGES_DIR", tmp_path / "images")
    return sent


def figures(docs):
    return [d for d in docs if d.metadata["content_type"] == "figure_caption"]


def one_page_pdf(tmp_path, draw) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    draw(page)
    path = tmp_path / "page.pdf"
    doc.save(str(path))
    return path


# --------------------------------------------------------------------------
# the two regression cases, through load_pdf
# --------------------------------------------------------------------------
def test_pew_p16_has_no_embedded_image_yet_gets_a_figure_caption(tmp_path, captured):
    path = fixture_pdf(tmp_path, "pew_p16_bar_chart")
    with pymupdf.open(path) as doc:
        assert doc[0].get_images(full=True) == []  # the old logic has nothing to grab
    found = figures(pdf_loader.load_pdf(path))
    assert len(found) == 1 and found[0].metadata["page"] == 1
    assert found[0].metadata["figure_source"] == "vector_render"
    assert Path(found[0].metadata["image_path"]).exists()
    assert len(captured) == 1


def test_itc_p9_pie_and_bar_chart_each_get_a_figure_caption(tmp_path, captured):
    path = fixture_pdf(tmp_path, "itc_p9_pie_and_bars")
    with pymupdf.open(path) as doc:
        assert sum(len(p.get_images(full=True)) for p in doc) == 0  # zero raster images at all
    found = figures(pdf_loader.load_pdf(path))
    assert len(found) == 2 and {d.metadata["page"] for d in found} == {1}
    assert all(d.metadata["figure_source"] == "vector_render" for d in found)
    assert len(captured) == 2


def test_the_crops_that_reach_the_captioner_are_real_images_above_the_size_floor(tmp_path, captured):
    pdf_loader.load_pdf(fixture_pdf(tmp_path, "itc_p9_pie_and_bars"))
    for png in captured:
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        pix = pymupdf.Pixmap(png)
        assert min(pix.width, pix.height) >= 150 and max(pix.width, pix.height) <= vc.RENDER_MAX_SIDE_PX


@pytest.mark.parametrize("name,page_number,expected", [
    ("PG_2020.03.09_US-Germany_FINAL.pdf", 16, 1),
    ("f86d073b0d735ac873a65d906ba82758.pdf", 9, 2),
])
def test_the_real_documents_when_available(name, page_number, expected, captured):
    folders = [Path(os.environ["VERIRAG_TEST_PDF_DIR"])] if os.environ.get("VERIRAG_TEST_PDF_DIR") else []
    folders += [REPO / "eval_store" / "test_pdfs"]
    path = next((f / name for f in folders if (f / name).exists()), None)
    if path is None:
        pytest.skip(f"{name} not available locally (set VERIRAG_TEST_PDF_DIR)")
    found = [d for d in figures(pdf_loader.load_pdf(path)) if d.metadata["page"] == page_number]
    assert len(found) == expected
    assert all(d.metadata["figure_source"] == "vector_render" for d in found)


# --------------------------------------------------------------------------
# what must NOT be turned into a figure
# --------------------------------------------------------------------------
def test_a_banded_table_is_not_a_chart(tmp_path, captured):
    def draw(page):
        for row in range(24):  # shaded row bands + hairline rules, like a financial table
            y = 100 + row * 14
            shape = page.new_shape()
            shape.draw_rect(pymupdf.Rect(60, y, 520, y + 13))
            shape.finish(fill=(0.88, 0.93, 0.98) if row % 2 else (1, 1, 1), color=None)
            shape.commit()
            line = page.new_shape()
            line.draw_line((60, y + 13.5), (520, y + 13.5))
            line.finish(color=(0.5, 0.5, 0.5), width=0.4)
            line.commit()
            page.insert_text((64, y + 10), f"Item {row}", fontsize=8)
            page.insert_text((400, y + 10), f"{1000 + row * 37:,}", fontsize=8)
    assert figures(pdf_loader.load_pdf(one_page_pdf(tmp_path, draw))) == []
    assert captured == []


def test_text_only_page_is_untouched(tmp_path, captured):
    docs = pdf_loader.load_pdf(one_page_pdf(tmp_path, lambda p: p.insert_textbox(
        pymupdf.Rect(72, 72, 500, 700), "Plain prose. " * 120, fontsize=11)))
    assert figures(docs) == [] and captured == []


def test_a_chart_already_covered_by_a_captioned_raster_is_not_captioned_twice(tmp_path, captured):
    def draw(page):
        draw_bar_chart(page)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 250), False)
        pix.clear_with(235)
        page.insert_image(pymupdf.Rect(40, 80, 280, 250), pixmap=pix)  # a captionable raster over the chart
    docs = pdf_loader.load_pdf(one_page_pdf(tmp_path, draw))
    found = figures(docs)
    assert len(found) == 1 and "figure_source" not in found[0].metadata  # the raster path, once
    assert len(captured) == 1


def test_a_chart_under_only_a_tiny_raster_is_still_captured(tmp_path, captured):
    # the pipeline does not caption an image under 150 px, so that image covers nothing
    def draw(page):
        draw_bar_chart(page)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), False)
        pix.clear_with(200)
        page.insert_image(pymupdf.Rect(40, 80, 280, 250), pixmap=pix)
    found = figures(pdf_loader.load_pdf(one_page_pdf(tmp_path, draw)))
    assert [d.metadata.get("figure_source") for d in found] == ["vector_render"]


def test_shaded_boxes_without_numeric_labels_are_not_a_chart(tmp_path, captured):
    assert_spoiler_blocks(tmp_path, spoiled=lambda p: draw_bar_chart(p, labels=False),
                          clean=lambda p: draw_bar_chart(p))


def test_a_text_heavy_region_is_not_a_chart(tmp_path, captured):
    def spoiled(page):
        draw_bar_chart(page)
        page.insert_textbox(pymupdf.Rect(40, 90, 290, 235), "word " * 400, fontsize=5)  # >150 words inside
    assert_spoiler_blocks(tmp_path, spoiled=spoiled, clean=lambda p: draw_bar_chart(p))


def test_an_undersized_chart_is_ignored(tmp_path, captured):
    assert_spoiler_blocks(tmp_path, spoiled=lambda p: draw_bar_chart(p, scale=0.4),
                          clean=lambda p: draw_bar_chart(p))


def test_the_region_size_rule_itself_rejects_a_small_chart():
    # at the detector, not downstream: load_pdf's 150 px floor would also drop a small
    # crop, which would let this rule go untested
    def page_with(scale):
        page = pymupdf.open().new_page(width=595, height=842)
        draw_bar_chart(page, scale=scale)
        return page

    assert len(vc.find_vector_charts(page_with(1.0), min_raster_px=150)) == 1
    assert vc.find_vector_charts(page_with(0.4), min_raster_px=150) == []
    small = vc.find_vector_charts(page_with(0.4), min_raster_px=150)
    assert small == [] and vc.MIN_WIDTH_PT == 120 and vc.MIN_HEIGHT_PT == 70


def test_table_style_hairline_rules_disqualify_a_region(tmp_path, captured):
    # banded tables draw their rules as thin RECTANGLES (the 10-K filings show dozens per
    # table); the same chart with 12 of them is a table, not a chart
    def spoiled(page):
        draw_bar_chart(page)
        for k in range(12):
            rule = page.new_shape()
            rule.draw_rect(pymupdf.Rect(55, 105 + k * 11, 265, 105.8 + k * 11))
            rule.finish(fill=(0.6, 0.6, 0.6), color=None)
            rule.commit()
    assert_spoiler_blocks(tmp_path, spoiled=spoiled, clean=lambda p: draw_bar_chart(p))


# --------------------------------------------------------------------------
# bounds and robustness
# --------------------------------------------------------------------------
def test_at_most_six_vector_charts_per_page(tmp_path, captured):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=2400)
    for i in range(8):
        draw_bar_chart(page, x=60, y=20 + 280 * i)
    path = tmp_path / "many.pdf"
    doc.save(str(path))
    assert len(figures(pdf_loader.load_pdf(path))) == vc.MAX_PER_PAGE == 6


def test_two_charts_on_one_page_get_distinct_image_files(tmp_path, captured):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    draw_bar_chart(page, y=60)
    draw_bar_chart(page, y=360)
    path = tmp_path / "two.pdf"
    doc.save(str(path))
    found = figures(pdf_loader.load_pdf(path))
    assert len({d.metadata["image_path"] for d in found}) == 2


def test_a_page_that_cannot_be_analysed_yields_nothing_instead_of_failing():
    class Broken:
        number = 0

        def get_drawings(self):
            raise RuntimeError("corrupt page")

    assert vc.find_vector_charts(Broken(), min_raster_px=150) == []


def test_a_failed_caption_skips_the_chart_without_aborting_the_document(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_loader, "caption_image_with_status", lambda b: None)
    monkeypatch.setattr(pdf_loader, "IMAGES_DIR", tmp_path / "images")
    assert figures(pdf_loader.load_pdf(fixture_pdf(tmp_path, "pew_p16_bar_chart"))) == []


def test_render_is_padded_clipped_to_the_page_and_capped(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    png, width, height = vc.render_region(page, pymupdf.Rect(0, 0, 595, 842))  # whole page
    assert max(width, height) <= vc.RENDER_MAX_SIDE_PX
    png, width, height = vc.render_region(page, pymupdf.Rect(100, 100, 220, 170))  # the smallest chart
    assert min(width, height) >= 150
