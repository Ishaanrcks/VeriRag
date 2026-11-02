"""Gate 2's visual re-check of caption-only claims.

The caption never certifies; the figure's pixels may. These tests stub the vision
call, so none of them touches a model or the network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pymupdf
import pytest

from verirag.critic import gate2_groundedness as g2
from verirag.critic import visual_check as vc
from verirag.generation.generator import Citation, GeneratedAnswer
from verirag.generation.partial import VERIFIED_VERDICTS
from verirag.retrieval.types import SearchHit

CLAIM = 'The pie chart titled "GLOBAL iOS BREAKDOWN" labels iOS 9 at 51%.'


@pytest.fixture(autouse=True)
def _clear_cache():
    vc._CACHE.clear()
    yield
    vc._CACHE.clear()


def png_bytes(shade=200, size=(240, 180)):
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, *size), False)
    pix.clear_with(shade)
    return pix.tobytes("png")


def caption_hit(tmp_path, chunk_id="cap1", with_image=True, page=3, pdf=None):
    meta = {"content_type": "figure_caption", "page": page}
    if with_image:
        p = tmp_path / f"{chunk_id}.png"
        p.write_bytes(png_bytes())
        meta["image_path"] = str(p)
    if pdf is not None:
        meta["source_path"] = str(pdf)
    return SearchHit(chunk_id=chunk_id, raw_text="A pie chart of iOS versions.", metadata=meta)


def stub_ask(monkeypatch, verdicts):
    """Make the vision call return ``verdicts`` in order; record the claims asked."""
    asked = []
    queue = list(verdicts)

    def fake(data, claim):
        asked.append(claim)
        return queue.pop(0) if queue else "CANNOT_TELL"

    monkeypatch.setattr(vc, "_ask", fake)
    return asked


# --------------------------------------------------------------------------
# the checker itself
# --------------------------------------------------------------------------
def test_yes_certifies_and_no_and_cannot_tell_do_not(monkeypatch, tmp_path):
    for verdict, expected in (("YES", "YES"), ("NO", "NO"), ("CANNOT_TELL", "CANNOT_TELL")):
        vc._CACHE.clear()
        stub_ask(monkeypatch, [verdict])
        assert vc.VisualChecker().check(CLAIM, caption_hit(tmp_path)) == expected


def test_a_missing_image_falls_back_to_rendering_the_page(monkeypatch, tmp_path):
    doc = pymupdf.open()
    for _ in range(4):
        doc.new_page(width=300, height=300).insert_text((40, 60), "iOS 9  51%", fontsize=14)
    pdf = tmp_path / "doc.pdf"
    doc.save(str(pdf))

    seen = {}

    def fake(data, claim):
        seen["bytes"] = data
        return "YES"

    monkeypatch.setattr(vc, "_ask", fake)
    hit = caption_hit(tmp_path, with_image=False, page=3, pdf=pdf)
    assert vc.VisualChecker().check(CLAIM, hit) == "YES"
    assert seen["bytes"][:8] == b"\x89PNG\r\n\x1a\n"  # a real render reached the model


def test_an_image_path_that_no_longer_exists_also_falls_back(monkeypatch, tmp_path):
    doc = pymupdf.open()
    doc.new_page(width=200, height=200)
    pdf = tmp_path / "d.pdf"
    doc.save(str(pdf))
    hit = caption_hit(tmp_path, with_image=True, page=1, pdf=pdf)
    (tmp_path / "cap1.png").unlink()  # the saved figure is gone
    stub_ask(monkeypatch, ["YES"])
    assert vc.VisualChecker().check(CLAIM, hit) == "YES"


def test_no_image_and_no_pdf_yields_no_judgement(monkeypatch, tmp_path):
    stub_ask(monkeypatch, ["YES"])
    checker = vc.VisualChecker()
    assert checker.check(CLAIM, caption_hit(tmp_path, with_image=False)) is None
    assert checker.checks[0]["verdict" if isinstance(checker.checks[0], dict) else "verdict"] if False else True
    assert checker.checks[0].verdict == "error:no_image"


def test_the_budget_caps_the_number_of_calls(monkeypatch, tmp_path):
    asked = stub_ask(monkeypatch, ["NO"] * 50)
    checker = vc.VisualChecker(budget=6)
    # 8 DIFFERENT figures, so the cache cannot absorb them
    for i in range(8):
        hit = caption_hit(tmp_path, chunk_id=f"c{i}")
        (tmp_path / f"c{i}.png").write_bytes(png_bytes(shade=10 * i))
        checker.check(f"{CLAIM} variant {i}", hit)
    assert len(asked) == 6 and checker.used == 6
    assert [c.verdict for c in checker.checks][-2:] == ["error:budget_exhausted"] * 2


def test_a_repeated_image_and_claim_is_cached_and_costs_no_budget(monkeypatch, tmp_path):
    asked = stub_ask(monkeypatch, ["YES"])
    checker = vc.VisualChecker(budget=1)
    hit = caption_hit(tmp_path)
    assert [checker.check(CLAIM, hit) for _ in range(4)] == ["YES"] * 4
    assert len(asked) == 1 and checker.used == 1
    assert sum(1 for c in checker.checks if c.verdict.startswith("cached:")) == 3


def test_every_check_is_logged_with_claim_image_and_verdict(monkeypatch, tmp_path):
    stub_ask(monkeypatch, ["NO"])
    checker = vc.VisualChecker()
    checker.check(CLAIM, caption_hit(tmp_path))
    entry = checker.checks[0]
    assert entry.verdict == "NO" and entry.chunk_id == "cap1"
    assert CLAIM[:40] in entry.claim and entry.image.endswith("cap1.png")


def test_a_vision_error_is_not_a_certification(monkeypatch, tmp_path):
    monkeypatch.setattr(vc, "_ask", lambda d, c: "error:APIError")
    checker = vc.VisualChecker()
    assert checker.check(CLAIM, caption_hit(tmp_path)) is None


def test_the_feature_can_be_switched_off(monkeypatch, tmp_path):
    asked = stub_ask(monkeypatch, ["YES"])
    monkeypatch.setattr(vc.settings, "gate2_visual_check_enabled", False)
    assert vc.VisualChecker().check(CLAIM, caption_hit(tmp_path)) is None
    assert asked == []


# --------------------------------------------------------------------------
# strict parsing of the model's reply
# --------------------------------------------------------------------------
@pytest.mark.parametrize("reply,expected", [
    ("YES", "YES"), ("yes", "YES"), ("YES.", "YES"),
    ("NO", "NO"), ("no", "NO"),
    ("CANNOT_TELL", "CANNOT_TELL"),
    ("The image is too blurry. CANNOT_TELL", "CANNOT_TELL"),
    ("Maybe, it looks like 51%", "CANNOT_TELL"),   # anything unparseable fails closed
    ("", "CANNOT_TELL"),
])
def test_reply_parsing_fails_closed(monkeypatch, reply, expected):
    message = SimpleNamespace(content=reply)
    completions = SimpleNamespace(create=lambda **kw: SimpleNamespace(
        choices=[SimpleNamespace(message=message)]))
    monkeypatch.setattr(vc, "_ask", vc._ask)
    import verirag.loaders.captioning as cap
    monkeypatch.setattr(cap, "_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=completions)))
    assert vc._ask(png_bytes(), CLAIM) == expected


def test_the_prompt_demands_one_word_and_visible_evidence():
    p = " ".join(vc._PROMPT.split())
    assert "exactly one word" in p and "CANNOT_TELL" in p
    assert "Judge ONLY what is visible" in p and "must actually be readable" in p


# --------------------------------------------------------------------------
# through Gate 2
# --------------------------------------------------------------------------
def _run_gate2(monkeypatch, hits, claim_text, content_types=None):
    payload = json.dumps([dict(text=claim_text, claim_type="factual", cited_numbers=[1])])
    import verirag.critic.claims as claims_mod
    monkeypatch.setattr(claims_mod, "chat_completion", lambda **kw: payload)
    monkeypatch.setattr(g2, "score_claim_against_premises",
                        lambda c, premises: [SimpleNamespace(entailment_prob=0.99) for _ in premises])
    monkeypatch.setattr(g2, "check_entailment_windowed",
                        lambda premise, c: SimpleNamespace(entailment_prob=0.99))
    citation = Citation(number=1, parent_text="", source_path="/d/x.pdf", doc_id="d", pages=[3],
                        contributing_chunk_ids=[h.chunk_id for h in hits],
                        content_types=sorted({h.metadata["content_type"] for h in hits}))
    answer = GeneratedAnswer(answer_text=claim_text, citations=[citation],
                             format_used=None, hits_used=hits)
    return g2.check_groundedness(answer)


def test_gate2_certifies_a_caption_only_claim_the_figure_confirms(monkeypatch, tmp_path):
    stub_ask(monkeypatch, ["YES"])
    verdict = _run_gate2(monkeypatch, [caption_hit(tmp_path)], CLAIM)
    assert [c.verdict for c in verdict.claims] == ["visually_verified"]
    assert verdict.overall_grounded and verdict.unsupported_claims == []
    assert verdict.visual_checks and verdict.visual_checks[0]["verdict"] == "YES"


@pytest.mark.parametrize("reply", ["NO", "CANNOT_TELL"])
def test_gate2_keeps_rejecting_when_the_figure_does_not_confirm(monkeypatch, tmp_path, reply):
    stub_ask(monkeypatch, [reply])
    verdict = _run_gate2(monkeypatch, [caption_hit(tmp_path)], CLAIM)
    assert [c.verdict for c in verdict.claims] == ["caption_only_uncertified"]
    assert not verdict.overall_grounded
    assert verdict.claims[0].figure_reference  # the reader is still pointed at the figure


def test_a_text_backed_claim_never_triggers_a_visual_check(monkeypatch, tmp_path):
    asked = stub_ask(monkeypatch, ["YES"])
    text_hit = SearchHit(chunk_id="t1", raw_text="iOS 9 is at 51%.",
                         metadata={"content_type": "text", "page": 3})
    verdict = _run_gate2(monkeypatch, [text_hit], CLAIM)
    assert [c.verdict for c in verdict.claims] == ["supported_single"]
    assert asked == [] and verdict.visual_checks == []


def test_visually_verified_counts_as_grounded_everywhere():
    from verirag.pipeline_log import claim_verdict_bucket

    assert "visually_verified" in VERIFIED_VERDICTS           # partial-answer facts list
    assert "visually_verified" not in g2._BLOCKING_VERDICTS   # does not block grounding
    assert claim_verdict_bucket("visually_verified") == "grounded"
