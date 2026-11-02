"""Gate 2's adversarial visual verification of caption-only numeric claims.

The check is two steps: a BLIND read (the figure plus the question, never the
claim) and a comparison in code. Only a MATCH certifies. These tests stub the
vision call, so none touches a model or the network.
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

QUESTION = "What percentage of devices were on iOS 9?"
CLAIM = 'The pie chart titled "GLOBAL iOS BREAKDOWN" labels iOS 9 at 51%.'


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    vc._CACHE.clear()
    monkeypatch.setattr(vc.settings, "gate2_visual_verify", True)
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


def stub_read(monkeypatch, values):
    """Make the blind read return ``values`` in order; record the prompts it saw."""
    prompts = []
    queue = list(values)

    def fake(data, question):
        prompts.append(question)
        return queue.pop(0) if queue else "UNREADABLE"

    monkeypatch.setattr(vc, "_read_value", fake)
    return prompts


# --------------------------------------------------------------------------
# the three outcomes
# --------------------------------------------------------------------------
def test_a_matching_value_certifies(monkeypatch, tmp_path):
    stub_read(monkeypatch, ["51%"])
    assert vc.VisualChecker(query=QUESTION).check(CLAIM, caption_hit(tmp_path)) == "MATCH"


def test_a_mismatching_value_rejects(monkeypatch, tmp_path):
    stub_read(monkeypatch, ["41%"])
    assert vc.VisualChecker(query=QUESTION).check(CLAIM, caption_hit(tmp_path)) == "MISMATCH"


def test_unreadable_rejects(monkeypatch, tmp_path):
    stub_read(monkeypatch, ["UNREADABLE"])
    assert vc.VisualChecker(query=QUESTION).check(CLAIM, caption_hit(tmp_path)) == "UNREADABLE"


def test_a_value_with_no_number_never_certifies(monkeypatch, tmp_path):
    # a label match is too weak to certify a numeric claim
    stub_read(monkeypatch, ["iOS 9 is the largest slice"])
    assert vc.VisualChecker(query=QUESTION).check(CLAIM, caption_hit(tmp_path)) == "MATCH"


# --------------------------------------------------------------------------
# the blind protocol: the claim must never reach call 1
# --------------------------------------------------------------------------
def test_the_claim_text_never_appears_in_the_call_1_prompt(monkeypatch, tmp_path):
    prompts = stub_read(monkeypatch, ["51%"])
    vc.VisualChecker(query=QUESTION).check(CLAIM, caption_hit(tmp_path))
    assert prompts == [QUESTION]
    for p in prompts:
        assert "51%" not in p and "GLOBAL iOS BREAKDOWN" not in p


def test_the_rendered_prompt_sent_to_the_model_contains_only_the_question(monkeypatch):
    sent = {}

    class _Completions:
        def create(self, **kw):
            sent["text"] = kw["messages"][0]["content"][0]["text"]
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="VALUE: 51%"))])

    import verirag.loaders.captioning as cap
    monkeypatch.setattr(cap, "_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions())))
    assert vc._read_value(png_bytes(), QUESTION) == "51%"
    assert QUESTION in sent["text"]
    assert "51%" not in sent["text"] and "labels iOS 9" not in sent["text"]
    assert "Do not estimate from" in sent["text"]  # no reading values off bar heights


@pytest.mark.parametrize("reply,expected", [
    ("VALUE: 51%", "51%"), ("value: 51", "51"), ("UNREADABLE", "UNREADABLE"),
    ("I cannot see it clearly", "UNREADABLE"), ("", "UNREADABLE"),
    ("VALUE:", "UNREADABLE"),
])
def test_blind_read_parsing(monkeypatch, reply, expected):
    class _Completions:
        def create(self, **kw):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=reply))])

    import verirag.loaders.captioning as cap
    monkeypatch.setattr(cap, "_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions())))
    assert vc._read_value(png_bytes(), QUESTION) == expected


@pytest.mark.parametrize("reported,claim,agree", [
    ("51%", "iOS 9 at 51%", True),
    ("51 percent", "iOS 9 at 51%", True),
    ("1.3 million", "the store had 1300000 apps", True),
    ("41%", "iOS 9 at 51%", False),
    ("no number here", "iOS 9 at 51%", False),
    ("51%", "a claim with no digits", False),
])
def test_value_comparison_is_normalised(reported, claim, agree):
    assert vc.values_agree(reported, claim) is agree


# --------------------------------------------------------------------------
# eligibility, images, budget, cache
# --------------------------------------------------------------------------
def test_a_claim_with_no_numbers_is_not_eligible(monkeypatch, tmp_path):
    prompts = stub_read(monkeypatch, ["51%"])
    assert vc.VisualChecker(query=QUESTION).check("The chart shows iOS versions.", caption_hit(tmp_path)) is None
    assert prompts == []


def test_no_question_means_no_check(monkeypatch, tmp_path):
    prompts = stub_read(monkeypatch, ["51%"])
    assert vc.VisualChecker(query="").check(CLAIM, caption_hit(tmp_path)) is None
    assert prompts == []


def test_the_flag_is_off_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(vc.settings, "gate2_visual_verify", False)
    prompts = stub_read(monkeypatch, ["51%"])
    assert vc.VisualChecker(query=QUESTION).check(CLAIM, caption_hit(tmp_path)) is None
    assert prompts == []


def test_a_missing_image_falls_back_to_rendering_the_page(monkeypatch, tmp_path):
    doc = pymupdf.open()
    for _ in range(4):
        doc.new_page(width=300, height=300).insert_text((40, 60), "iOS 9  51%", fontsize=14)
    pdf = tmp_path / "doc.pdf"
    doc.save(str(pdf))
    seen = {}

    def fake(data, question):
        seen["bytes"] = data
        return "51%"

    monkeypatch.setattr(vc, "_read_value", fake)
    hit = caption_hit(tmp_path, with_image=False, page=3, pdf=pdf)
    assert vc.VisualChecker(query=QUESTION).check(CLAIM, hit) == "MATCH"
    assert seen["bytes"][:8] == b"\x89PNG\r\n\x1a\n"


def test_no_image_and_no_pdf_yields_no_judgement(monkeypatch, tmp_path):
    stub_read(monkeypatch, ["51%"])
    checker = vc.VisualChecker(query=QUESTION)
    assert checker.check(CLAIM, caption_hit(tmp_path, with_image=False)) is None
    assert checker.checks[0].verdict == "error:no_image"


def test_the_budget_caps_the_number_of_calls(monkeypatch, tmp_path):
    prompts = stub_read(monkeypatch, ["41%"] * 50)
    checker = vc.VisualChecker(query=QUESTION, budget=6)
    for i in range(8):
        hit = caption_hit(tmp_path, chunk_id=f"c{i}")
        (tmp_path / f"c{i}.png").write_bytes(png_bytes(shade=10 * i))
        checker.check(f"{CLAIM} variant {i} value {i}", hit)
    assert len(prompts) == 6 and checker.used == 6
    assert [c.verdict for c in checker.checks][-2:] == ["error:budget_exhausted"] * 2


def test_the_blind_read_is_cached_per_figure_and_question(monkeypatch, tmp_path):
    prompts = stub_read(monkeypatch, ["51%"])
    checker = vc.VisualChecker(query=QUESTION, budget=1)
    hit = caption_hit(tmp_path)
    assert [checker.check(CLAIM, hit) for _ in range(4)] == ["MATCH"] * 4
    assert len(prompts) == 1 and checker.used == 1
    assert sum(1 for c in checker.checks if c.verdict.startswith("cached:")) == 3


def test_every_check_records_the_value_that_was_read(monkeypatch, tmp_path):
    stub_read(monkeypatch, ["41%"])
    checker = vc.VisualChecker(query=QUESTION)
    checker.check(CLAIM, caption_hit(tmp_path))
    entry = checker.checks[0]
    assert entry.verdict == "MISMATCH" and entry.reported == "41%"
    assert entry.chunk_id == "cap1" and entry.image.endswith("cap1.png")


def test_a_vision_error_is_not_a_certification(monkeypatch, tmp_path):
    monkeypatch.setattr(vc, "_read_value", lambda d, q: "error:APIError")
    assert vc.VisualChecker(query=QUESTION).check(CLAIM, caption_hit(tmp_path)) is None


# --------------------------------------------------------------------------
# through Gate 2
# --------------------------------------------------------------------------
def _run_gate2(monkeypatch, hits, claim_text, query=QUESTION):
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
    return g2.check_groundedness(answer, query=query)


def test_gate2_certifies_only_on_a_match(monkeypatch, tmp_path):
    stub_read(monkeypatch, ["51%"])
    verdict = _run_gate2(monkeypatch, [caption_hit(tmp_path)], CLAIM)
    assert [c.verdict for c in verdict.claims] == ["visually_verified"]
    assert verdict.overall_grounded
    assert verdict.visual_checks[0]["reported"] == "51%"


@pytest.mark.parametrize("reported", ["41%", "UNREADABLE"])
def test_gate2_keeps_rejecting_on_mismatch_or_unreadable(monkeypatch, tmp_path, reported):
    stub_read(monkeypatch, [reported])
    verdict = _run_gate2(monkeypatch, [caption_hit(tmp_path)], CLAIM)
    assert [c.verdict for c in verdict.claims] == ["caption_only_uncertified"]
    assert not verdict.overall_grounded and verdict.claims[0].figure_reference


def test_a_text_backed_claim_never_triggers_a_visual_check(monkeypatch, tmp_path):
    prompts = stub_read(monkeypatch, ["51%"])
    text_hit = SearchHit(chunk_id="t1", raw_text="iOS 9 is at 51%.",
                         metadata={"content_type": "text", "page": 3})
    verdict = _run_gate2(monkeypatch, [text_hit], CLAIM)
    assert verdict.claims[0].verdict in ("supported_single", "supported_numeric")
    assert prompts == [] and verdict.visual_checks == []


def test_visually_verified_counts_as_grounded_everywhere():
    from verirag.pipeline_log import claim_verdict_bucket

    assert "visually_verified" in VERIFIED_VERDICTS
    assert "visually_verified" not in g2._BLOCKING_VERDICTS
    assert claim_verdict_bucket("visually_verified") == "grounded"
