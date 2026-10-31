"""Strict vs lenient scoring of partial answers, and the harness refusal phrase.

STRICT (the headline) lets the extractor read only the Verified facts of a partial
answer; LENIENT reads the whole response and is an upper bound that can credit a
statement the system left unverified. No model: the extractor is a deterministic
stub that "finds" a value only if it appears in the text it is given.
"""

from __future__ import annotations

import pytest

from verirag.eval.harness import classify_outcome
from verirag.eval.mmlongbench_scorer import (
    LENIENT_LABEL,
    STRICT_LABEL,
    pooled_metrics,
    score_response_dual,
    strict_input,
)
from verirag.generation.partial import VerifiedFact, render_partial_answer

QUESTION = "What percentage of West Germans view the two relationships as equally important?"
SAMPLE = dict(question=QUESTION, answer="29%", answer_format="Float")


class StubExtractor:
    """Returns the first candidate value that occurs in the text it is shown."""

    def __init__(self, candidates=("29%", "21%", "43%")):
        self.candidates, self.seen = candidates, []

    def __call__(self, question, text):
        self.seen.append(text)
        for value in self.candidates:
            if value in text:
                return f"Extracted answer: {value}\nAnswer format: Float"
        return "Extracted answer: Not answerable\nAnswer format: String"


def _partial(body, *fact_texts):
    return render_partial_answer(body, [VerifiedFact(text=t, citation_numbers=[1]) for t in fact_texts])


# --------------------------------------------------------------------------
# #8's shape: the value is in the body, NOT in the Verified facts
# --------------------------------------------------------------------------
BODY_ONLY_VALUE = _partial(
    'The chart shows 29% of West Germans volunteered "Both". The passages do not say what that means.',
    "The chart's note says that Don't know responses are not shown.",
)


def test_unverified_value_scores_zero_strict_and_correct_lenient():
    out = score_response_dual(SAMPLE, BODY_ONLY_VALUE, StubExtractor())
    assert out["partial"] is True
    assert (out["strict"]["pred"], out["strict"]["score"]) == ("Not answerable", 0.0)
    assert (out["lenient"]["pred"], out["lenient"]["score"]) == ("29%", 1.0)


def test_strict_extractor_never_sees_the_body():
    extractor = StubExtractor()
    score_response_dual(SAMPLE, BODY_ONLY_VALUE, extractor)
    lenient_text, strict_text = extractor.seen
    assert "volunteered" in lenient_text  # lenient reads the whole response
    assert "volunteered" not in strict_text and "29%" not in strict_text
    assert "Don't know responses are not shown" in strict_text


# --------------------------------------------------------------------------
# #6's shape: the value IS in the Verified facts
# --------------------------------------------------------------------------
def test_value_in_the_verified_facts_is_correct_under_both():
    response = _partial("A hedged opening. Some figure is mentioned.",
                        'The chart lists West Germany at 29% for "Both (VOL)".')
    out = score_response_dual(SAMPLE, response, StubExtractor())
    assert out["strict"]["score"] == 1.0 and out["lenient"]["score"] == 1.0
    assert out["strict"]["pred"] == out["lenient"]["pred"] == "29%"


# --------------------------------------------------------------------------
# a fully verified answer has no section: the two passes coincide
# --------------------------------------------------------------------------
def test_fully_verified_answer_is_scored_once_and_identically():
    extractor = StubExtractor()
    out = score_response_dual(SAMPLE, "West Germans: 29% chose Both. [1]", extractor)
    assert out["partial"] is False
    assert out["strict"] is out["lenient"] and out["strict"]["score"] == 1.0
    assert len(extractor.seen) == 1  # no second extractor call


def test_strict_input_is_only_the_facts_for_a_partial_answer():
    assert "Don't know" in strict_input(BODY_ONLY_VALUE)
    assert "29%" not in strict_input(BODY_ONLY_VALUE)
    assert strict_input("Plain answer. [1]") == "Plain answer. [1]"


# --------------------------------------------------------------------------
# pooled metrics, both ways, and the labels that keep them apart
# --------------------------------------------------------------------------
def test_pooled_metrics_strict_is_below_lenient_when_a_value_was_unverified():
    extractor = StubExtractor()
    responses = [BODY_ONLY_VALUE, _partial("Opening.", "West Germany is at 29% for Both."),
                 "Not enough to answer."]
    samples = [SAMPLE, SAMPLE, dict(SAMPLE, answer="Not answerable", answer_format="None")]
    duals = [score_response_dual(s, r, extractor) for s, r in zip(samples, responses)]
    strict = pooled_metrics([d["strict"] for d in duals])
    lenient = pooled_metrics([d["lenient"] for d in duals])
    assert strict["n"] == lenient["n"] == 3
    assert strict["accuracy"] == pytest.approx(2 / 3)   # #1 scores 0, #2 and the decline score 1
    assert lenient["accuracy"] == pytest.approx(1.0)
    assert strict["recall"] == pytest.approx(0.5) and lenient["recall"] == pytest.approx(1.0)


def test_the_two_numbers_are_labelled_so_they_cannot_be_confused():
    assert "headline" in STRICT_LABEL.lower() and "STRICT" in STRICT_LABEL
    assert "upper bound" in LENIENT_LABEL and "unverified" in LENIENT_LABEL


# --------------------------------------------------------------------------
# harness: a decline phrased "do not give" is a decline
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    # the openings of Queensland Q51 and Q54, which the harness scored as answers
    "The passages do not give a GPU-hour figure for training the 34M G-reasoner model. "
    "They state that the GFM is trained with 16 A100 GPUs for 10 epochs [1].",
    "The passages do not give a numeric learning rate for training QpiGNN. "
    "Algorithm 1 lists a learning rate as an input [1].",
])
def test_do_not_give_opening_is_classified_as_a_refusal(text):
    assert classify_outcome(False, text) == "refused"


def test_a_substantive_answer_that_mentions_giving_later_is_still_an_answer():
    # only the FIRST sentence is checked, so a passing negation does not demote it
    assert classify_outcome(False, "GFM-RAG has 8M parameters [1]. The passages do not give a cost.") \
        == "answered"
