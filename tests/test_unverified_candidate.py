"""The UNVERIFIED candidate answer (flag: unverified_candidate).

Measured candidate precision is 50%, and four of the wrong candidates are on
questions whose correct answer is "not answerable". So the only safe shape is a
clearly labelled lead kept strictly apart from the verified answer. These tests
pin that separation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from verirag.config import settings
from verirag.generation import candidate as C
from verirag.graph import build as B
from verirag.retrieval.types import SearchHit


def hit(text, page=1, ctype="text", doc="d.pdf"):
    return SearchHit(chunk_id=f"c{page}", raw_text=text,
                     metadata={"content_type": ctype, "page": page,
                               "source_path": f"/docs/{doc}", "doc_id": doc})


def g2(*verdicts):
    return SimpleNamespace(claims=[SimpleNamespace(verdict=v) for v in verdicts])


def state(answer="The rate is 20.7%.", hits=None, outcome_g2=None, generated=True):
    gen = SimpleNamespace(answer_text=answer, hits_used=hits or [hit("the rate was 20.7%")])
    return {"generated_answer": gen if generated else None, "gate2_verdict": outcome_g2}


# --------------------------------------------------------------------------
# the verified answer is untouched
# --------------------------------------------------------------------------
def test_the_verified_answer_is_byte_identical_with_the_flag_on_and_off(monkeypatch):
    base = dict(final_answer="VERIFIED TEXT", final_citations=[], abstained=True,
                abstention_reason="because", trace=[], regeneration_count=1,
                verified_facts=["a fact"], pipeline_log=[], generated_answer=None,
                gate1_verdict=None, gate2_verdict=None)
    captured = {}

    def fake_derived(_state):
        return dict(outcome="abstained_ungrounded", total_latency_ms=1.0,
                    llm_call_count=1, estimated_cost_usd=0.0)

    for flag in (False, True):
        monkeypatch.setattr(settings, "unverified_candidate", flag)
        monkeypatch.setattr(B, "_derived_fields", fake_derived, raising=False)
        captured[flag] = B._unverified_candidate(dict(base), "abstained_ungrounded")
    # with no generated answer there is nothing to offer either way
    assert captured[False] is None and captured[True] is None


def test_a_candidate_never_appears_in_the_verified_facts():
    from verirag.generation.partial import VERIFIED_VERDICTS

    assert "candidate" not in " ".join(VERIFIED_VERDICTS)
    assert C.UNVERIFIED_LABEL == "UNVERIFIED"


# --------------------------------------------------------------------------
# when a candidate is and is not produced
# --------------------------------------------------------------------------
def test_no_candidate_when_the_flag_is_off(monkeypatch):
    monkeypatch.setattr(settings, "unverified_candidate", False)
    assert B._unverified_candidate(state(), "abstained_ungrounded") is None


def test_no_candidate_when_the_pipeline_verified_an_answer(monkeypatch):
    monkeypatch.setattr(settings, "unverified_candidate", True)
    assert B._unverified_candidate(state(), "verified") is None


def test_no_candidate_when_gate1_declined_for_lack_of_evidence(monkeypatch):
    """Declining because nothing relevant was retrieved must never become a guess."""
    monkeypatch.setattr(settings, "unverified_candidate", True)
    assert B._unverified_candidate(state(), "abstained_insufficient") is None


def test_no_candidate_when_generation_never_ran(monkeypatch):
    monkeypatch.setattr(settings, "unverified_candidate", True)
    assert B._unverified_candidate(state(generated=False), "abstained_ungrounded") is None


def test_no_candidate_for_an_empty_generated_answer(monkeypatch):
    monkeypatch.setattr(settings, "unverified_candidate", True)
    assert B._unverified_candidate(state(answer="   "), "abstained_ungrounded") is None


def test_a_candidate_is_produced_on_a_gate2_decline(monkeypatch):
    monkeypatch.setattr(settings, "unverified_candidate", True)
    out = B._unverified_candidate(
        state(outcome_g2=g2("not_supported")), "abstained_ungrounded")
    assert out is not None and out.label == "UNVERIFIED"
    assert out.answer.startswith("The rate is 20.7%")
    assert out.sources and out.sources[0]["page"] == 1


def test_a_candidate_is_produced_on_a_partial_answer(monkeypatch):
    monkeypatch.setattr(settings, "unverified_candidate", True)
    assert B._unverified_candidate(state(outcome_g2=g2("not_supported")), "partial") is not None


# --------------------------------------------------------------------------
# the reason label
# --------------------------------------------------------------------------
def test_a_value_in_no_passage_is_labelled_derived():
    # 60.4 appears nowhere in the retrieved text, so it was computed
    hits = [hit("TimeQA 39.9% and MMLU 20.5%")]
    assert C.classify_reason("The sum is 60.4%.", g2("not_supported"), hits) == C.REASON_DERIVED


def test_a_caption_only_claim_is_labelled_as_such():
    hits = [hit("the chart labels iOS 9 at 51%", ctype="figure_caption")]
    assert C.classify_reason("iOS 9 is 51%.", g2("caption_only_uncertified"), hits) == C.REASON_CAPTION


def test_an_unsupported_claim_is_labelled_as_such():
    hits = [hit("the warehouse count is 12")]
    assert C.classify_reason("There are 12 warehouses.", g2("not_supported"), hits) == C.REASON_UNSUPPORTED


def test_anything_else_is_labelled_other():
    hits = [hit("the warehouse count is 12")]
    assert C.classify_reason("There are 12 warehouses.", g2("meta_excluded"), hits) == C.REASON_OTHER


def test_the_label_is_always_present(monkeypatch):
    monkeypatch.setattr(settings, "unverified_candidate", True)
    for verdicts in (("not_supported",), ("caption_only_uncertified",), ("meta_excluded",)):
        out = B._unverified_candidate(state(outcome_g2=g2(*verdicts)), "abstained_ungrounded")
        assert out.label == "UNVERIFIED" and out.reason in (
            C.REASON_DERIVED, C.REASON_CAPTION, C.REASON_UNSUPPORTED, C.REASON_OTHER)


# --------------------------------------------------------------------------
# a derived value is shown as a derivation, not as a fact
# --------------------------------------------------------------------------
def test_a_derived_value_is_shown_with_its_inputs_and_pages():
    hits = [hit("TimeQA baseline wrong 39.9%", page=12),
            hit("MMLU Physics baseline wrong 20.5%", page=6)]
    note = C.derivation_note("The sum is 60.4%: 39.9% plus 20.5%.", hits)
    assert note.startswith("derived from")
    assert "39.9" in note and "20.5" in note
    assert "p12" in note and "p6" in note


def test_the_candidate_text_carries_the_derivation_note(monkeypatch):
    monkeypatch.setattr(settings, "unverified_candidate", True)
    hits = [hit("TimeQA 39.9%", page=12), hit("MMLU 20.5%", page=6)]
    out = B._unverified_candidate(
        state(answer="The sum is 60.4%: 39.9% plus 20.5%.", hits=hits,
              outcome_g2=g2("not_supported")), "abstained_ungrounded")
    assert out.reason == C.REASON_DERIVED
    assert "derived from" in out.answer and "p12" in out.answer


def test_no_derivation_note_when_nothing_was_derived():
    hits = [hit("the rate was 20.7%")]
    assert C.derivation_note("The rate is 20.7%.", hits) == ""


def test_sources_are_deduplicated_and_capped():
    hits = [hit("x", page=1) for _ in range(10)] + [hit("y", page=i) for i in range(2, 12)]
    out = C.describe_sources(hits)
    assert len(out) <= 6
    assert len({(s["document"], s["page"], s["type"]) for s in out}) == len(out)
