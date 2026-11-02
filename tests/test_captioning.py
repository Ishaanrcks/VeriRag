"""Caption prompt content and finish_reason (truncation) handling.

The vision call is mocked: no network, no model.
"""

from __future__ import annotations

import pymupdf
import pytest

from verirag.loaders import captioning as C
from verirag.loaders import pdf as pdf_loader


def _script(monkeypatch, responses):
    """Make ``_caption_with_retry`` return ``responses`` in order; record max_tokens."""
    calls: list[int] = []
    queue = list(responses)

    def fake(data_url, max_tokens=C._CAPTION_MAX_TOKENS):
        calls.append(max_tokens)
        return queue.pop(0)

    monkeypatch.setattr(C, "_caption_with_retry", fake)
    return calls


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------
def test_prompt_asks_for_plotted_values():
    prompt = C._CAPTION_PROMPT
    assert "numeric values" in prompt
    assert "MUST state" in prompt
    assert "rather than only describing the trend" in prompt
    # an invented number is worse than a missing one
    assert "do not invent" in prompt.lower()


def test_prompt_is_what_the_vision_call_sends(monkeypatch):
    sent = {}

    class _Completions:
        def create(self, **kwargs):
            sent.update(kwargs)
            message = type("M", (), {"content": "ok"})()
            return type("R", (), {"choices": [type("C", (), {"message": message,
                                                              "finish_reason": "stop"})()]})()

    client = type("Cl", (), {"chat": type("Ch", (), {"completions": _Completions()})()})()
    monkeypatch.setattr(C, "_client", lambda: client)
    content, finish = C._caption_with_retry.__wrapped__("data:image/png;base64,AAAA", 123)
    assert (content, finish) == ("ok", "stop")
    text_part = sent["messages"][0]["content"][0]["text"]
    assert text_part == C._CAPTION_PROMPT and "numeric values" in text_part
    assert sent["max_tokens"] == 123


# --------------------------------------------------------------------------
# finish_reason handling
# --------------------------------------------------------------------------
def test_complete_caption_makes_one_call_and_is_not_flagged(monkeypatch):
    calls = _script(monkeypatch, [("A complete caption.", "stop")])
    result = C.caption_image_with_status(b"\x89PNG\r\n\x1a\nxxxx")
    assert result == C.Caption("A complete caption.", truncated=False)
    assert calls == [C._CAPTION_MAX_TOKENS]


def test_truncated_caption_is_retried_with_more_tokens(monkeypatch):
    calls = _script(monkeypatch, [("Cut off mid-sen", "length"), ("The full caption.", "stop")])
    result = C.caption_image_with_status(b"img")
    assert result == C.Caption("The full caption.", truncated=False)
    assert calls == [C._CAPTION_MAX_TOKENS, C._CAPTION_RETRY_MAX_TOKENS]
    assert C._CAPTION_RETRY_MAX_TOKENS > C._CAPTION_MAX_TOKENS


def test_still_truncated_after_retry_is_flagged_not_dropped(monkeypatch, caplog):
    calls = _script(monkeypatch, [("part one", "length"), ("part one and two", "length")])
    with caplog.at_level("WARNING", logger=C.logger.name):
        result = C.caption_image_with_status(b"img")
    assert result == C.Caption("part one and two", truncated=True)
    assert len(calls) == 2  # retried exactly once
    assert any("still truncated" in r.message for r in caplog.records)


def test_length_with_empty_content_then_success(monkeypatch):
    # the reasoning model can spend the whole budget thinking and emit nothing
    _script(monkeypatch, [("", "length"), ("Now it fits.", "stop")])
    assert C.caption_image_with_status(b"img") == C.Caption("Now it fits.", False)


def test_empty_after_retry_returns_none(monkeypatch):
    _script(monkeypatch, [(None, "length"), (None, "length")])
    assert C.caption_image_with_status(b"img") is None


def test_api_error_returns_none(monkeypatch):
    def boom(data_url, max_tokens=0):
        raise RuntimeError("bad request")

    monkeypatch.setattr(C, "_caption_with_retry", boom)
    assert C.caption_image_with_status(b"img") is None


def test_caption_image_wrapper_returns_text_only(monkeypatch):
    _script(monkeypatch, [("text", "stop")])
    assert C.caption_image(b"img") == "text"


# --------------------------------------------------------------------------
# the flag reaches the chunk metadata
# --------------------------------------------------------------------------
def _pdf_with_image(path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A page with a figure.")
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 220, 220), False)
    pix.clear_with(180)
    page.insert_image(pymupdf.Rect(72, 100, 292, 320), pixmap=pix)
    doc.save(str(path))
    return path


@pytest.mark.parametrize("truncated", [True, False])
def test_load_pdf_flags_truncated_caption_in_metadata(monkeypatch, tmp_path, truncated):
    monkeypatch.setattr(pdf_loader, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(
        pdf_loader, "caption_image_with_status",
        lambda _bytes: C.Caption("some caption", truncated=truncated),
    )
    docs = pdf_loader.load_pdf(_pdf_with_image(tmp_path / "fig.pdf"))
    figures = [d for d in docs if d.metadata["content_type"] == "figure_caption"]
    assert len(figures) == 1
    if truncated:
        assert figures[0].metadata["caption_truncated"] is True
    else:
        assert "caption_truncated" not in figures[0].metadata


# --------------------------------------------------------------------------
# images in formats the vision API rejects are re-encoded first
# --------------------------------------------------------------------------
def _pdf_with_png(tmp_path):
    """A one-page PDF holding a real embedded PNG, and its xref."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 200), False)
    pix.clear_with(128)
    page.insert_image(pymupdf.Rect(50, 50, 250, 250), pixmap=pix)
    path = tmp_path / "img.pdf"
    doc.save(str(path))
    opened = pymupdf.open(path)
    return opened, opened[0].get_images(full=True)[0][0]


@pytest.mark.parametrize("ext", ["png", "jpeg", "jpg", "webp", "gif"])
def test_an_already_supported_format_is_passed_through_untouched(tmp_path, ext):
    doc, xref = _pdf_with_png(tmp_path)
    extracted = dict(doc.extract_image(xref))
    extracted["ext"] = ext
    data, out_ext = pdf_loader._vision_safe_image(doc, xref, extracted)
    assert data is extracted["image"] and out_ext == ext  # identical object: no re-encode


@pytest.mark.parametrize("ext", ["jpx", "jp2", "tiff", "bmp", "jb2", "ccitt"])
def test_an_unsupported_format_is_re_encoded_to_png(tmp_path, ext):
    # the real corpus case is jpx (JPEG 2000); the API rejects anything outside
    # webp/png/jpeg/gif with a 400, which used to lose the figure silently
    doc, xref = _pdf_with_png(tmp_path)
    extracted = dict(doc.extract_image(xref))
    extracted["ext"] = ext
    data, out_ext = pdf_loader._vision_safe_image(doc, xref, extracted)
    assert out_ext == "png" and data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data is not extracted["image"]


def test_cmyk_in_a_supported_format_is_left_alone(tmp_path):
    # a CMYK JPEG was measured to be ACCEPTED and captioned correctly by the vision
    # API, so converting it only inflated the payload (~2.4x) for no gain
    doc, xref = _pdf_with_png(tmp_path)
    extracted = dict(doc.extract_image(xref))
    extracted["ext"] = "jpeg"
    extracted["colorspace"] = 4  # CMYK
    data, out_ext = pdf_loader._vision_safe_image(doc, xref, extracted)
    assert data is extracted["image"] and out_ext == "jpeg"


def test_a_failed_re_encode_falls_back_to_the_original_bytes(tmp_path):
    doc, xref = _pdf_with_png(tmp_path)
    extracted = dict(doc.extract_image(xref))
    extracted["ext"] = "jpx"
    data, out_ext = pdf_loader._vision_safe_image(doc, 999999, extracted)  # no such xref
    assert data is extracted["image"] and out_ext == "jpx"  # unchanged, never raises


def test_the_converted_bytes_are_what_reach_the_captioner(tmp_path, monkeypatch):
    """End to end: load_pdf must caption the CONVERTED image, not the raw bytes."""
    seen: list[bytes] = []

    def stub(image_bytes):
        seen.append(image_bytes)
        return C.Caption("[caption]")

    monkeypatch.setattr(pdf_loader, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(pdf_loader, "caption_image_with_status", stub)
    real = pdf_loader._vision_safe_image
    # force the conversion branch for every image in the document
    monkeypatch.setattr(pdf_loader, "_vision_safe_image",
                        lambda d, x, e: real(d, x, dict(e, ext="jpx")))
    doc, _ = _pdf_with_png(tmp_path)
    docs = pdf_loader.load_pdf(tmp_path / "img.pdf")
    assert [d.metadata["content_type"] for d in docs].count("figure_caption") == 1
    assert len(seen) == 1 and seen[0][:8] == b"\x89PNG\r\n\x1a\n"
