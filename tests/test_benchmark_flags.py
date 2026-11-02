"""The benchmark-only experiment flags.

Every flag defaults False, so production behaviour is exactly what it was before
they existed; each test turns on only the flag it is about.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from verirag.config import settings
from verirag.critic import gate1_coverage as g1
from verirag.critic import gate2_groundedness as g2
from verirag.eval.flags import FLAGS, active_flags, apply_flags
from verirag.generation import generator as gen
from verirag.retrieval.types import SearchHit


def hit(chunk_id, text, score=0.9, content_type="text", page=1):
    # rerank_score is a FIELD on SearchHit, not metadata: Gate 1 reads hit.rerank_score
    return SearchHit(chunk_id=chunk_id, raw_text=text, rerank_score=score,
                     metadata={"content_type": content_type, "page": page,
                               "source_path": "/d/x.pdf"})


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------
def test_all_flags_default_off_so_production_is_unchanged():
    assert [f for f in FLAGS if getattr(settings, f)] == []
    assert settings.generation_temperature == 0.2  # production value


def test_apply_flags_sets_only_what_is_named_and_pins_temperature(monkeypatch):
    for f in FLAGS:
        monkeypatch.setattr(settings, f, False)
    monkeypatch.setattr(settings, "generation_temperature", 0.2)
    assert apply_flags("gate1_advisory,gate2_numeric_match") == ["gate1_advisory", "gate2_numeric_match"]
    assert active_flags() == ["gate1_advisory", "gate2_numeric_match"]
    assert settings.generation_temperature == 0.0


def test_a_typo_in_a_flag_name_raises(monkeypatch):
    for f in FLAGS:
        monkeypatch.setattr(settings, f, False)
    with pytest.raises(ValueError, match="unknown benchmark flag"):
        apply_flags("gate1_advsory")


# --------------------------------------------------------------------------
# FLAG 1: gate1_advisory
# --------------------------------------------------------------------------
def test_advisory_gate1_passes_the_band_without_calling_the_judge(monkeypatch):
    called = []
    monkeypatch.setattr(g1, "_llm_judge_coverage",
                        lambda *a, **k: called.append(1) or ("insufficient", "no"))
    monkeypatch.setattr(settings, "gate1_advisory", True)
    # 0.10 sits inside the escalation band (low 0.05 .. high 0.15)
    verdict = g1.check_coverage("q", [hit("c1", "text", 0.10)])
    assert verdict.verdict == "sufficient" and verdict.decision_path == "threshold_high"
    assert called == []
    assert verdict.max_rerank_score == pytest.approx(0.10)  # score still recorded


def test_advisory_gate1_still_declines_below_the_floor(monkeypatch):
    monkeypatch.setattr(settings, "gate1_advisory", True)
    verdict = g1.check_coverage("q", [hit("c1", "text", 0.01)])
    assert verdict.verdict == "insufficient" and verdict.decision_path == "threshold_low"


def test_advisory_gate1_still_declines_with_no_hits(monkeypatch):
    monkeypatch.setattr(settings, "gate1_advisory", True)
    assert g1.check_coverage("q", []).verdict == "insufficient"


def test_without_the_flag_the_band_still_consults_the_judge(monkeypatch):
    called = []

    def judge(*a, **k):
        called.append(1)
        return "insufficient", "not enough"

    monkeypatch.setattr(g1, "_llm_judge_coverage", judge)
    verdict = g1.check_coverage("q", [hit("c1", "text", 0.10)])
    assert called == [1] and verdict.decision_path == "llm_escalation"


# --------------------------------------------------------------------------
# FLAG 3: gate2_numeric_match (and the number normaliser it shares)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected_subset", [
    ("20.7%", {"20.7"}),
    ("20.7 percent", {"20.7"}),
    ("$1,234", {"1234"}),
    ("1.3 million", {"1300000"}),
    ("2 billion", {"2000000000"}),
    ("no numbers here", set()),
])
def test_number_normalisation(text, expected_subset):
    assert expected_subset <= g2.normalised_numbers(text)


def test_numeric_match_needs_the_number_and_the_subject(monkeypatch):
    monkeypatch.setattr(settings, "gate2_numeric_match", True)
    claim = "The effective tax rate for fiscal 2023 is 20.7%."
    # right number, right subject
    assert g2.numeric_support(claim, [("c1", "Effective tax rate fiscal 2023 20.7%")])
    # right number, unrelated subject -> must not verify
    assert g2.numeric_support(claim, [("c2", "Page 20.7 of the appendix lists warehouses")]) is None
    # right subject, wrong number
    assert g2.numeric_support(claim, [("c3", "Effective tax rate fiscal 2023 was 19.0%")]) is None


def test_numeric_match_rejects_a_chunk_with_the_right_numbers_about_something_else():
    """Isolates the entity requirement.

    The earlier test passed for the wrong reason: its distractor chunk was missing
    the claim's year, so the NUMBER check rejected it and removing the entity check
    changed nothing. Here every number in the claim IS present, so only the entity
    overlap can reject it.
    """
    claim = "The effective tax rate for fiscal 2023 is 20.7%."
    same_numbers_wrong_subject = [
        ("c1", "In 2023 the company opened 20.7 new distribution centres per region.")
    ]
    assert set(g2.normalised_numbers(claim)) <= set(
        g2.normalised_numbers(same_numbers_wrong_subject[0][1])
    )  # the numbers really are all there
    assert g2.numeric_support(claim, same_numbers_wrong_subject) is None


def test_a_claim_with_no_numbers_is_not_a_numeric_match():
    assert g2.numeric_support("The company operates warehouses.", [("c1", "warehouses")]) is None


# --------------------------------------------------------------------------
# FLAG 2: gate2_citation_repair
# --------------------------------------------------------------------------
def test_citation_repair_finds_support_the_citation_failed_to_point_at(monkeypatch):
    monkeypatch.setattr(settings, "gate2_citation_repair", True)
    hits = [hit("c1", "Unrelated prose about distribution centres."),
            hit("c2", "The effective tax rate for fiscal 2023 was 20.7%.")]
    answer = SimpleNamespace(hits_used=hits)
    claim = SimpleNamespace(text="The effective tax rate for fiscal 2023 is 20.7%.")
    found = g2.repair_citation(claim, answer, {h.chunk_id: h.raw_text for h in hits})
    assert [c for c, _ in found] == ["c2"]


def test_citation_repair_returns_nothing_when_no_chunk_supports_the_claim(monkeypatch):
    monkeypatch.setattr(settings, "gate2_citation_repair", True)
    hits = [hit("c1", "Distribution centres in Ohio and Texas.")]
    answer = SimpleNamespace(hits_used=hits)
    claim = SimpleNamespace(text="The effective tax rate for fiscal 2023 is 20.7%.")
    assert g2.repair_citation(claim, answer, {"c1": hits[0].raw_text}) == []


def test_an_unresolvable_claim_is_still_rejected_without_the_flag(monkeypatch):
    payload = json.dumps([dict(text="The tax rate is 20.7%.", claim_type="factual",
                               cited_numbers=[9])])  # citation 9 is never defined
    import verirag.critic.claims as claims_mod
    monkeypatch.setattr(claims_mod, "chat_completion", lambda **kw: payload)
    hits = [hit("c1", "The effective tax rate for fiscal 2023 was 20.7%.")]
    citation = gen.Citation(number=1, parent_text="", source_path="/d/x.pdf", doc_id="d",
                            pages=[1], contributing_chunk_ids=["c1"], content_types=["text"])
    answer = gen.GeneratedAnswer(answer_text="The tax rate is 20.7%.", citations=[citation],
                                 format_used=None, hits_used=hits)
    verdict = g2.check_groundedness(answer)
    assert verdict.claims[0].verdict == "not_supported"
    assert verdict.citation_repairs == []


# --------------------------------------------------------------------------
# FLAG 5: refusal_requires_search
# --------------------------------------------------------------------------
@pytest.mark.parametrize("query,expected", [
    ("How many weeks does the project run?", True),
    ("What percentage of devices used iOS 9?", True),
    ("Who is the chief executive?", False),          # not a numeric question
    ("How many weeks does the project run?", True),
])
def test_candidate_value_detection(query, expected):
    hits = [hit("c1", "The project runs over 16 weeks and carries 8 units.")]
    assert gen.has_candidate_value(query, hits) is expected


def test_no_candidate_when_the_passages_hold_no_numbers():
    hits = [hit("c1", "The project runs for several weeks.")]
    assert gen.has_candidate_value("How many weeks does the project run?", hits) is False


def test_a_refusal_is_regenerated_when_a_candidate_exists(monkeypatch):
    monkeypatch.setattr(settings, "refusal_requires_search", True)
    replies = iter(["Cannot be determined from the provided sources. The passages do not say.",
                    "16 weeks [1]. The Field Service Project runs over 16 weeks."])
    prompts = []

    def fake(messages, **kw):
        prompts.append(messages[0]["content"])
        return next(replies)

    monkeypatch.setattr(gen, "chat_completion", fake)
    hits = [hit("c1", "The Field Service Project runs over 16 weeks and carries 8 Units.")]
    out = gen.generate_answer("How many weeks to reach the milestone?", hits, gen.AnswerFormat.CONCISE)
    assert out.answer_text.startswith("16 weeks")
    assert len(prompts) == 2
    assert "DO contain candidate values" in prompts[1]


def test_a_refusal_stands_when_no_candidate_exists(monkeypatch):
    monkeypatch.setattr(settings, "refusal_requires_search", True)
    calls = []

    def fake(messages, **kw):
        calls.append(1)
        return "Cannot be determined from the provided sources. Nothing in the passages applies."

    monkeypatch.setattr(gen, "chat_completion", fake)
    hits = [hit("c1", "The project runs for several weeks, duration unspecified.")]
    out = gen.generate_answer("How many weeks to reach the milestone?", hits, gen.AnswerFormat.CONCISE)
    assert out.answer_text.startswith("Cannot be determined")
    assert len(calls) == 1  # no retry


def test_a_refusal_stands_for_a_non_numeric_question(monkeypatch):
    monkeypatch.setattr(settings, "refusal_requires_search", True)
    calls = []
    monkeypatch.setattr(gen, "chat_completion",
                        lambda messages, **kw: calls.append(1) or
                        "Cannot be determined from the provided sources.")
    hits = [hit("c1", "Revenue was 412 million in 2021.")]
    out = gen.generate_answer("Who signed the agreement?", hits, gen.AnswerFormat.CONCISE)
    assert out.answer_text.startswith("Cannot be determined") and len(calls) == 1


def test_without_the_flag_a_refusal_is_never_regenerated(monkeypatch):
    calls = []
    monkeypatch.setattr(gen, "chat_completion",
                        lambda messages, **kw: calls.append(1) or
                        "Cannot be determined from the provided sources.")
    hits = [hit("c1", "The Field Service Project runs over 16 weeks.")]
    out = gen.generate_answer("How many weeks?", hits, gen.AnswerFormat.CONCISE)
    assert out.answer_text.startswith("Cannot be determined") and len(calls) == 1


# --------------------------------------------------------------------------
# sharding
# --------------------------------------------------------------------------
def test_shards_are_disjoint_and_cover_everything():
    ids = list(range(1, 89))
    shards = [ids[i - 1::4] for i in range(1, 5)]
    assert sorted(x for s in shards for x in s) == ids
    assert all(len(set(s)) == len(s) for s in shards)
    assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 1
