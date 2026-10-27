"""Gate 2 provenance tier on the COMBINATION path.

A claim supported only by model-generated figure captions must not be certified.
The single-chunk path always enforced this; the combination path (no single
chunk entails the claim, but the chunks jointly do) used to certify it as
``supported_combination``.

Most tests stub the NLI calls so they exercise only the branch logic and need no
model. One test uses the real NLI checkpoint (skipped if it cannot be loaded).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from verirag.critic import gate2_groundedness as g2
from verirag.critic.claims import Claim
from verirag.generation.generator import AnswerFormat, Citation, GeneratedAnswer
from verirag.retrieval.types import SearchHit


def _check(texts, ctypes, claim_text="a claim"):
    """Run the real check_groundedness on one claim citing all given chunks."""
    hits = [
        SearchHit(chunk_id=f"c{i}", raw_text=text, metadata={"content_type": ctype})
        for i, (text, ctype) in enumerate(zip(texts, ctypes))
    ]
    citation = Citation(
        number=1, parent_text="", source_path="/x/p.pdf", doc_id="d", pages=[1],
        contributing_chunk_ids=[h.chunk_id for h in hits],
        content_types=sorted(set(ctypes)),
    )
    answer = GeneratedAnswer(
        answer_text="x [1]", citations=[citation],
        format_used=AnswerFormat.CONCISE, hits_used=hits,
    )
    return answer, claim_text


@pytest.fixture
def stub_nli(monkeypatch):
    """No single chunk entails (unless it contains SOLO); a premise containing
    BOTH ALPHA and BETA entails jointly."""
    def per_chunk(claim, premises):
        return [SimpleNamespace(entailment_prob=0.99 if "SOLO" in p else 0.0)
                for p in premises]

    def windowed(premise, claim):
        both = "ALPHA" in premise and "BETA" in premise
        return SimpleNamespace(entailment_prob=0.99 if both else 0.0)

    monkeypatch.setattr(g2, "score_claim_against_premises", per_chunk)
    monkeypatch.setattr(g2, "check_entailment_windowed", windowed)


def _verdict(monkeypatch, answer, claim_text):
    monkeypatch.setattr(
        g2, "extract_claims",
        lambda a: [Claim(text=claim_text, claim_type="factual", cited_numbers=[1])],
    )
    result = g2.check_groundedness(answer)
    return result, result.claims[0]


def test_two_captions_that_jointly_entail_are_not_certified(stub_nli, monkeypatch):
    answer, claim = _check(["ALPHA fact", "BETA fact"], ["figure_caption"] * 2)
    result, v = _verdict(monkeypatch, answer, claim)
    assert v.verdict == "caption_only_uncertified"  # was supported_combination
    assert v.evidence_chunk_ids == ["c0", "c1"]
    assert v.figure_reference  # points the reader at the figure instead
    assert result.overall_grounded is False  # blocking, like the single path


def test_single_caption_path_is_unchanged(stub_nli, monkeypatch):
    answer, claim = _check(["SOLO caption", "other"], ["figure_caption"] * 2)
    _, v = _verdict(monkeypatch, answer, claim)
    assert v.verdict == "caption_only_uncertified"
    assert v.evidence_chunk_ids == ["c0"]


def test_text_only_combination_is_still_certified(stub_nli, monkeypatch):
    answer, claim = _check(["ALPHA fact", "BETA fact"], ["text", "text"])
    result, v = _verdict(monkeypatch, answer, claim)
    assert v.verdict == "supported_combination"
    assert v.evidence_chunk_ids == ["c0", "c1"]
    assert result.overall_grounded is True


def test_combination_certified_by_text_alone_excludes_the_caption(stub_nli, monkeypatch):
    answer, claim = _check(["ALPHA fact", "BETA fact", "unrelated figure"],
                           ["text", "text", "figure_caption"])
    _, v = _verdict(monkeypatch, answer, claim)
    assert v.verdict == "supported_combination"
    # The caption did not decide the verdict, so it is not in the support set.
    assert v.evidence_chunk_ids == ["c0", "c1"]


def test_combination_that_needs_a_caption_is_not_certified(stub_nli, monkeypatch):
    answer, claim = _check(["ALPHA fact", "BETA fact"], ["text", "figure_caption"])
    _, v = _verdict(monkeypatch, answer, claim)
    assert v.verdict == "caption_only_uncertified"
    assert v.evidence_chunk_ids == ["c1"]  # the caption the claim depended on


# --- real NLI model: the exact failure observed in the wild -----------------
A = "Figure 3 compares model Alpha with model Beta using attention heatmaps."
B = "In that figure, every axis is ticked from 0 to 144, and the colorbar spans 0.2 to 1.0."
CLAIM = "The attention heatmaps comparing model Alpha with model Beta have axes ticked from 0 to 144."


@pytest.fixture
def real_nli():
    try:
        from verirag.critic.nli import check_entailment_windowed
        check_entailment_windowed("warm up", "warm up")
    except Exception as exc:  # noqa: BLE001 - model not available offline
        pytest.skip(f"NLI checkpoint unavailable: {exc}")


def test_real_nli_caption_combination_is_not_certified(real_nli, monkeypatch):
    # Neither chunk entails the claim alone; together they do. As captions this
    # was previously certified as supported_combination.
    answer, claim = _check([A, B], ["figure_caption"] * 2, CLAIM)
    _, v = _verdict(monkeypatch, answer, claim)
    assert v.verdict == "caption_only_uncertified"

    # Reference: the same evidence as extracted text is still certified.
    answer, claim = _check([A, B], ["text", "text"], CLAIM)
    _, v = _verdict(monkeypatch, answer, claim)
    assert v.verdict == "supported_combination"
