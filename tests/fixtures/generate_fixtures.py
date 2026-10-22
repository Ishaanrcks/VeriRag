"""Generate small synthetic fixture files for exercising the loaders.

Run from anywhere:

    python tests/fixtures/generate_fixtures.py

Produces, next to this file:
- sample.txt, sample.md   : a couple of paragraphs each
- sample.docx             : 3 paragraphs (python-docx)
- sample.pptx             : 2 slides, title + body (python-pptx)
- sample.pdf              : 2 pages of real text, with a matplotlib bar chart
                            embedded as an image on page 2

Nothing here imports verirag.config, so no API key is needed to regenerate
fixtures. Only the PDF's figure-caption path (at load time) calls DeepSeek.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pymupdf  # noqa: E402
from docx import Document as DocxDocument  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.util import Pt  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent

TXT_BODY = (
    "VeriRAG is a self-verifying document question-answering system. It "
    "combines retrieval-augmented generation with a two-gate verification "
    "critic that checks answers against retrieved evidence.\n\n"
    "This plain-text fixture exists so the .txt loader can be exercised "
    "end to end without a real corpus. It contains two short paragraphs of "
    "ordinary prose."
)

MD_BODY = (
    "# VeriRAG Sample\n\n"
    "This is a **Markdown** fixture with a couple of paragraphs.\n\n"
    "- Retrieval grounds the answer in sources.\n"
    "- A critic verifies the answer before it is returned.\n\n"
    "The second paragraph exists only to give the loader some real text to "
    "read back during Phase 1 testing.\n"
)

DOCX_PARAGRAPHS = [
    "VeriRAG DOCX Fixture",
    "This Word document has three short paragraphs. The loader should "
    "concatenate them into a single text Document with no page number.",
    "Image handling for .docx is intentionally out of scope in the Phase 1 "
    "MVP; only paragraph text is extracted here.",
]

PDF_PAGE1 = (
    "VeriRAG PDF Fixture - Page 1\n\n"
    "This is the first page of a two-page synthetic PDF. It contains only "
    "real extractable text so the PDF text path can be verified. The second "
    "page additionally embeds a bar chart image, which exercises the "
    "figure-captioning path through the DeepSeek vision model.\n\n"
    "Retrieval-augmented generation retrieves relevant passages and then "
    "conditions the language model's answer on them."
)

PDF_PAGE2_TEXT = (
    "VeriRAG PDF Fixture - Page 2\n\n"
    "Below this text is an embedded bar chart. The loader should extract the "
    "page text as one Document and the chart as a separate figure_caption "
    "Document once the vision model describes it."
)


def _make_chart_png() -> bytes:
    """Render a simple labeled bar chart to PNG bytes (~600x450 px)."""
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    revenue = [23, 45, 31, 52]
    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    ax.bar(quarters, revenue, color="steelblue")
    ax.set_xlabel("Fiscal Quarter (2025)")
    ax.set_ylabel("Revenue (USD millions)")
    ax.set_title("Quarterly Revenue 2025")
    for x, y in zip(quarters, revenue):
        ax.text(x, y + 1, str(y), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def write_txt() -> None:
    (FIXTURES_DIR / "sample.txt").write_text(TXT_BODY, encoding="utf-8")


def write_md() -> None:
    (FIXTURES_DIR / "sample.md").write_text(MD_BODY, encoding="utf-8")


def write_docx() -> None:
    doc = DocxDocument()
    for para in DOCX_PARAGRAPHS:
        doc.add_paragraph(para)
    doc.save(str(FIXTURES_DIR / "sample.docx"))


def write_pptx() -> None:
    prs = Presentation()
    title_body_layout = prs.slide_layouts[1]  # "Title and Content"
    slides = [
        ("Introducing VeriRAG",
         "A self-verifying document QA system.\n"
         "Retrieval plus a two-gate verification critic."),
        ("How Verification Works",
         "Gate 1 checks support in retrieved evidence.\n"
         "Gate 2 checks the answer actually addresses the question."),
    ]
    for title_text, body_text in slides:
        slide = prs.slides.add_slide(title_body_layout)
        slide.shapes.title.text = title_text
        body = slide.placeholders[1]
        body.text = body_text
        for paragraph in body.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(18)
    prs.save(str(FIXTURES_DIR / "sample.pptx"))


def write_pdf() -> None:
    chart_png = _make_chart_png()
    doc = pymupdf.open()  # empty PDF

    page1 = doc.new_page()
    page1.insert_textbox(
        pymupdf.Rect(56, 56, 540, 760), PDF_PAGE1, fontsize=12, fontname="helv"
    )

    page2 = doc.new_page()
    page2.insert_textbox(
        pymupdf.Rect(56, 56, 540, 260), PDF_PAGE2_TEXT, fontsize=12, fontname="helv"
    )
    page2.insert_image(pymupdf.Rect(80, 280, 520, 620), stream=chart_png)

    doc.save(str(FIXTURES_DIR / "sample.pdf"))
    doc.close()


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    write_txt()
    write_md()
    write_docx()
    write_pptx()
    write_pdf()
    print(f"Fixtures written to {FIXTURES_DIR}")
    for name in sorted(p.name for p in FIXTURES_DIR.glob("sample.*")):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
