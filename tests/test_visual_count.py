"""The two-read visual counting protocol (flag: visual_count).

Both reads are blind: neither prompt may contain a claim or a candidate answer, and
a value is accepted only when two differently-worded reads of two different images
agree. These tests stub the vision call, so none touches a model or the network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pymupdf
import pytest

from verirag.config import settings
from verirag.critic import gate2_groundedness as g2
from verirag.critic import visual_count as vc
from verirag.generation.generator import Citation, GeneratedAnswer
from verirag.generation.partial import VERIFIED_VERDICTS
from verirag.retrieval.types import SearchHit

QUESTION = "How many blue arrows are part of the DISCIPLINED AGILE logo?"


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    vc._CACHE.clear()
    monkeypatch.setattr(settings, "visual_count", True)
    yield
    vc._CACHE.clear()


def png(shade=200, size=(240, 180)):
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, *size), False)
    pix.clear_with(shade)
    return pix.tobytes("png")


def make_pdf(tmp_path, pages=3):
    """A PDF whose pages differ, so their renders differ.

    Identical pages render to identical bytes and the cache then serves the second
    and third figure for free -- correct behaviour, but it hides the budget.
    """
    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page(width=300, height=300).insert_text(
            (40, 60), f"figure number {i} " + "x" * (i * 7), fontsize=12)
    path = tmp_path / "doc.pdf"
    doc.save(str(path))
    return path


def fig_hit(tmp_path, chunk_id="f1", page=1, with_crop=True, pdf=None, ctype="figure_caption"):
    meta = {"content_type": ctype, "page": page,
            "source_path": str(pdf if pdf is not None else make_pdf(tmp_path))}
    if with_crop:
        p = tmp_path / f"{chunk_id}.png"
        p.write_bytes(png())
        meta["image_path"] = str(p)
    return SearchHit(chunk_id=chunk_id, raw_text="A logo with arrows.", metadata=meta)


def stub_ask(monkeypatch, replies):
    """Feed replies in order; record (prompt_variant, question) for each call."""
    calls = []
    queue = list(replies)

    def fake(data, question, variant):
        calls.append({"variant": variant, "question": question, "bytes": data})
        return queue.pop(0) if queue else "UNREADABLE"

    monkeypatch.setattr(vc, "ask", fake)
    return calls


# --------------------------------------------------------------------------
# agreement is the only thing that certifies
# --------------------------------------------------------------------------
def test_two_agreeing_reads_certify(monkeypatch, tmp_path):
    stub_ask(monkeypatch, ["COUNT: 2", "COUNT: 2"])
    out = vc.VisualCounter().count(QUESTION, [fig_hit(tmp_path)])
    assert out.agreed and out.value == "2" and out.page == 1
    assert len(out.reads) == 2 and [r.parsed for r in out.reads] == ["2", "2"]


def test_disagreeing_reads_do_not_certify(monkeypatch, tmp_path):
    stub_ask(monkeypatch, ["COUNT: 2", "COUNT: 3"])
    out = vc.VisualCounter().count(QUESTION, [fig_hit(tmp_path)])
    assert not out.agreed and out.value is None


@pytest.mark.parametrize("replies", [
    ["UNREADABLE", "COUNT: 2"],
    ["COUNT: 2", "UNREADABLE"],
    ["UNREADABLE", "UNREADABLE"],
])
def test_unreadable_never_certifies(monkeypatch, tmp_path, replies):
    stub_ask(monkeypatch, replies)
    out = vc.VisualCounter().count(QUESTION, [fig_hit(tmp_path)])
    assert not out.agreed and out.value is None


def test_zero_is_a_legal_agreed_answer(monkeypatch, tmp_path):
    # two target questions have a gold answer of 0, so this must not be treated
    # as "nothing found"
    stub_ask(monkeypatch, ["COUNT: 0", "COUNT: 0"])
    out = vc.VisualCounter().count(QUESTION, [fig_hit(tmp_path)])
    assert out.agreed and out.value == "0"


def test_both_prompts_tell_the_model_that_zero_is_allowed():
    assert "COUNT: 0" in vc._PROMPT_PAGE and "Zero is a correct" in vc._PROMPT_PAGE
    assert "COUNT: 0" in vc._PROMPT_CROP and "do not round up to one" in vc._PROMPT_CROP


# --------------------------------------------------------------------------
# the reads are blind, and use two different images and wordings
# --------------------------------------------------------------------------
def test_neither_prompt_contains_a_claim_or_candidate_answer(monkeypatch, tmp_path):
    calls = stub_ask(monkeypatch, ["COUNT: 2", "COUNT: 2"])
    vc.VisualCounter().count(QUESTION, [fig_hit(tmp_path)])
    for c in calls:
        assert c["question"] == QUESTION      # only the question is passed through
    for template in (vc._PROMPT_PAGE, vc._PROMPT_CROP):
        rendered = template.format(question=QUESTION)
        assert "2" not in rendered.replace("COUNT: 0", "")  # no candidate leaks in
        assert "claim" not in rendered.lower()


def test_the_two_reads_use_different_images_and_different_wordings(monkeypatch, tmp_path):
    calls = stub_ask(monkeypatch, ["COUNT: 2", "COUNT: 2"])
    vc.VisualCounter().count(QUESTION, [fig_hit(tmp_path)])
    assert [c["variant"] for c in calls] == ["page", "crop"]
    assert calls[0]["bytes"] != calls[1]["bytes"]
    assert vc._PROMPT_PAGE != vc._PROMPT_CROP


def test_a_missing_crop_falls_back_to_the_page_for_both_reads(monkeypatch, tmp_path):
    calls = stub_ask(monkeypatch, ["COUNT: 2", "COUNT: 2"])
    out = vc.VisualCounter().count(QUESTION, [fig_hit(tmp_path, with_crop=False)])
    assert out.agreed                                  # still possible
    assert calls[0]["bytes"] == calls[1]["bytes"]      # but weaker: one image
    assert any("no crop available" in r.source for r in out.reads)


# --------------------------------------------------------------------------
# the trigger
# --------------------------------------------------------------------------
@pytest.mark.parametrize("query,expected", [
    ("How many blue arrows are in the logo?", True),
    ("How many people are there?", True),
    ("What colour is the slice shown in the figure?", True),
    ("What is the title of the document?", False),
    ("Summarise the risk factors section.", False),
])
def test_trigger_requires_a_counting_or_reading_question(tmp_path, query, expected):
    assert vc.should_count(query, [fig_hit(tmp_path)]) is expected


def test_no_trigger_on_a_pure_text_question(tmp_path):
    text_hit = SearchHit(chunk_id="t1", raw_text="prose",
                         metadata={"content_type": "text", "page": 1})
    assert vc.should_count("How many weeks does it run?", [text_hit]) is False


def test_no_trigger_when_the_flag_is_off(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "visual_count", False)
    assert vc.should_count(QUESTION, [fig_hit(tmp_path)]) is False


# --------------------------------------------------------------------------
# budget, caching, page limit
# --------------------------------------------------------------------------
def test_at_most_three_pages_are_tried(monkeypatch, tmp_path):
    pdf = make_pdf(tmp_path, pages=6)
    hits = [fig_hit(tmp_path, chunk_id=f"f{i}", page=i, pdf=pdf) for i in range(1, 6)]
    assert len(vc.figure_targets(hits)) == vc.MAX_PAGES


def test_the_budget_caps_the_vision_calls(monkeypatch, tmp_path):
    calls = stub_ask(monkeypatch, ["COUNT: 1", "COUNT: 9"] * 10)
    pdf = make_pdf(tmp_path, pages=4)
    # distinct crops as well as distinct pages, so nothing is served from the cache
    hits = []
    for i in range(1, 4):
        h = fig_hit(tmp_path, chunk_id=f"f{i}", page=i, pdf=pdf)
        (tmp_path / f"f{i}.png").write_bytes(png(shade=20 * i))
        hits.append(h)
    counter = vc.VisualCounter(budget=3)
    counter.count(QUESTION, hits)
    assert len(calls) == 3 and counter.used == 3
    assert any(r.raw_reply == "error:budget_exhausted" for r in counter.reads)


def test_identical_pages_are_served_from_the_cache(monkeypatch, tmp_path):
    """Three figures on byte-identical pages cost two calls, not six."""
    calls = stub_ask(monkeypatch, ["COUNT: 1", "COUNT: 9"] * 10)
    doc = pymupdf.open()
    for _ in range(4):
        doc.new_page(width=300, height=300).insert_text((40, 60), "same", fontsize=12)
    pdf = tmp_path / "same.pdf"
    doc.save(str(pdf))
    hits = [fig_hit(tmp_path, chunk_id="f1", page=i, pdf=pdf) for i in range(1, 4)]
    counter = vc.VisualCounter(budget=10)
    counter.count(QUESTION, hits)
    assert counter.used == 2 and len(calls) == 2
    assert len(counter.reads) == 6          # all six reads still logged


def test_a_repeated_image_and_variant_is_cached(monkeypatch, tmp_path):
    calls = stub_ask(monkeypatch, ["COUNT: 2", "COUNT: 2"])
    hit = fig_hit(tmp_path)
    first = vc.VisualCounter().count(QUESTION, [hit])
    second = vc.VisualCounter().count(QUESTION, [hit])
    assert first.value == second.value == "2"
    assert len(calls) == 2            # the second question made no new call


def test_every_read_is_logged_with_source_variant_and_raw_reply(monkeypatch, tmp_path):
    stub_ask(monkeypatch, ["COUNT: 2", "COUNT: 3"])
    out = vc.VisualCounter().count(QUESTION, [fig_hit(tmp_path)])
    assert [r.prompt_variant for r in out.reads] == ["page", "crop"]
    assert [r.raw_reply for r in out.reads] == ["COUNT: 2", "COUNT: 3"]
    assert out.reads[0].source.startswith("page ") and out.reads[1].source.startswith("crop ")


# --------------------------------------------------------------------------
# parsing, failing closed
# --------------------------------------------------------------------------
def test_an_unreadable_line_stops_a_later_count_being_read():
    """A reply that says UNREADABLE and then volunteers a number must not certify.

    Without the early return the parser would skip the UNREADABLE line and read the
    COUNT, certifying a value the model had just disclaimed.
    """
    assert vc.parse_read("UNREADABLE\nCOUNT: 3") is None
    assert vc.parse_read("UNREADABLE - the detail is lost\nVALUE: 51%") is None
    # the other order still reads the value, since the model committed first
    assert vc.parse_read("COUNT: 3\nUNREADABLE") == "3"


@pytest.mark.parametrize("reply,expected", [
    ("COUNT: 2", "2"), ("count: 0", "0"), ("VALUE: 51%", "51"),
    ("VALUE: 1,3 million", "1.3"), ("COUNT: 2.", "2"),
    ("UNREADABLE", None), ("I think there are two", None), ("", None),
    ("error:APIError", None), ("COUNT:", None),
])
def test_read_parsing_fails_closed(reply, expected):
    assert vc.parse_read(reply) == expected


# --------------------------------------------------------------------------
# through Gate 2
# --------------------------------------------------------------------------
def _run_gate2(monkeypatch, hits, answer_text, query):
    payload = json.dumps([dict(text=answer_text, claim_type="factual", cited_numbers=[1])])
    import verirag.critic.claims as claims_mod
    monkeypatch.setattr(claims_mod, "chat_completion", lambda **kw: payload)
    monkeypatch.setattr(g2, "score_claim_against_premises",
                        lambda c, premises: [SimpleNamespace(entailment_prob=0.99) for _ in premises])
    monkeypatch.setattr(g2, "check_entailment_windowed",
                        lambda premise, c: SimpleNamespace(entailment_prob=0.99))
    citation = Citation(number=1, parent_text="", source_path="/d/x.pdf", doc_id="d", pages=[1],
                        contributing_chunk_ids=[h.chunk_id for h in hits],
                        content_types=sorted({h.metadata["content_type"] for h in hits}))
    answer = GeneratedAnswer(answer_text=answer_text, citations=[citation],
                             format_used=None, hits_used=hits)
    return g2.check_groundedness(answer, query=query)


def test_gate2_adds_a_visual_counted_fact_on_agreement(monkeypatch, tmp_path):
    stub_ask(monkeypatch, ["COUNT: 2", "COUNT: 2"])
    v = _run_gate2(monkeypatch, [fig_hit(tmp_path)], "The logo contains arrows.", QUESTION)
    counted = [c for c in v.claims if c.verdict == "visual_counted"]
    assert len(counted) == 1 and "2" in counted[0].claim_text
    assert v.count_reads and len(v.count_reads) == 2


def test_gate2_adds_nothing_on_disagreement(monkeypatch, tmp_path):
    stub_ask(monkeypatch, ["COUNT: 2", "COUNT: 5"])
    v = _run_gate2(monkeypatch, [fig_hit(tmp_path)], "The logo contains arrows.", QUESTION)
    assert [c for c in v.claims if c.verdict == "visual_counted"] == []
    assert len(v.count_reads) == 2       # still logged


def test_gate2_does_not_count_for_a_non_counting_question(monkeypatch, tmp_path):
    calls = stub_ask(monkeypatch, ["COUNT: 2", "COUNT: 2"])
    v = _run_gate2(monkeypatch, [fig_hit(tmp_path)], "The document is a report.",
                   "What is the title of the document?")
    assert [c for c in v.claims if c.verdict == "visual_counted"] == []
    assert calls == [] and v.count_reads == []


def test_visual_counted_counts_as_grounded_everywhere():
    from verirag.pipeline_log import claim_verdict_bucket

    assert "visual_counted" in VERIFIED_VERDICTS
    assert "visual_counted" not in g2._BLOCKING_VERDICTS
    assert claim_verdict_bucket("visual_counted") == "grounded"
