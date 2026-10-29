"""Offline tests for the vendored MMLongBench-Doc scorer and extractor.

No network and no model: the extraction client is a stub.
"""

from __future__ import annotations

import ast
import sys
from types import SimpleNamespace

import pytest

from verirag.eval.mmlongbench_scorer import (
    FAILED_TO_EXTRACT,
    _parse_list,
    eval_acc_and_f1,
    eval_score,
    extract_answer,
    load_extraction_prompt,
    parse_extraction,
    score_response,
)


# --------------------------------------------------------------------------
# per-format scoring
# --------------------------------------------------------------------------
def test_int_is_exact():
    assert eval_score("3", "3", "Int") == 1.0
    assert eval_score("3", "3.0", "Int") == 1.0
    assert eval_score("3", "4", "Int") == 0.0
    assert eval_score("3", "three", "Int") == 0.0


def test_int_truncates_float_predictions():
    # Upstream quirk, kept: int(float(pred)) truncates, so "3.7" matches key 3.
    # The paper describes integers as exact match.
    assert eval_score("3", "3.7", "Int") == 1.0


def test_float_one_percent_tolerance():
    assert eval_score("100.0", "100.9", "Float") == 1.0  # 0.9% off
    assert eval_score("100.0", "102.0", "Float") == 0.0  # 2% off
    assert eval_score("2.5", "abc", "Float") == 0.0


def test_float_percentage_handling():
    # A key of 18.29% is matched by "18.29", "18.29%" and "0.1829".
    assert eval_score("18.29%", "18.29", "Float") == 1.0
    assert eval_score("18.29%", "0.1829", "Float") == 1.0
    assert eval_score("18.29%", "18.29%", "Float") == 1.0
    # ...and a key of 5 is matched by 0.05 / 500: scale-insensitive by design.
    assert eval_score("5", "0.05", "Float") == 1.0
    assert eval_score("5", "500", "Float") == 1.0
    assert eval_score("18.29%", "19.5", "Float") == 0.0


def test_float_small_values_are_looser_than_one_percent():
    # Upstream quirk, kept: the rounding fallback compares to the coarser of the
    # two decimal precisions (at least 2), so 0.044 matches key 0.04 -- 10% off,
    # not the "1% relative tolerance" the paper describes.
    assert eval_score("0.04", "0.044", "Float") == 1.0


def test_str_anls_with_floor():
    assert eval_score("Operating Activities", "operating activities", "Str") == 1.0
    close = eval_score("Operating Activities", "Operations Activities", "Str")
    assert 0.5 < close < 1.0
    assert eval_score("cat", "dog", "Str") == 0.0
    # similarity exactly at the 0.5 threshold is zeroed ("<=", not "<")
    assert eval_score("abcd", "abxx", "Str") == 0.0
    assert eval_score("abcdef", "abcdxx", "Str") == pytest.approx(1 - 2 / 6)


def test_str_exact_match_patterns_bypass_anls():
    assert eval_score("2015-03-12", "2015-03-12", "Str") == 1.0
    assert eval_score("2015-03-12", "2015-03-13", "Str") == 0.0  # ANLS would give 0.9
    assert eval_score("page 5", "page 6", "Str") == 0.0


def test_list_is_all_or_nothing_on_length():
    assert eval_score("['a', 'b', 'c']", "['a', 'b']", "List") == 0.0
    assert eval_score("['a', 'b']", "['a', 'b', 'c']", "List") == 0.0


def test_list_order_insensitive_and_numeric_exact():
    assert eval_score("['red', 'blue']", "['blue', 'red']", "List") == 1.0
    assert eval_score("['1', '2']", "['2', '1']", "List") == 1.0
    assert eval_score("['1', '2']", "['1', '3']", "List") == 0.0


def test_list_string_items_use_minimum_anls():
    score = eval_score(
        "['alpha team', 'beta group']", "['alpha team', 'bata group']", "List"
    )
    assert 0.5 < score < 1.0  # min over items, not a mean or a count
    assert eval_score("['alpha team', 'beta group']", "['alpha team', 'zzzzzzzzzz']", "List") == 0.0


def test_list_unquoted_prediction_does_not_crash():
    # Upstream eval()s this and raises NameError; here it is parsed leniently.
    assert eval_score("['a', 'b']", "[a, b]", "List") == 1.0


# --------------------------------------------------------------------------
# unanswerable handling
# --------------------------------------------------------------------------
def test_correct_decline_scores_one_and_wrong_answer_zero():
    assert eval_score("Not answerable", "Not answerable", "None") == 1.0
    assert eval_score("Not answerable", "42", "None") == 0.0
    assert eval_score("Not answerable", "Fail to answer", "None") == 0.0
    assert eval_score("42", "Not answerable", "Int") == 0.0


def test_decline_counts_in_accuracy_but_not_in_f1_numerator():
    answerable = [
        dict(answer="a", pred="a", score=1.0),
        dict(answer="b", pred="x", score=0.0),
    ]
    acc_before, f1_before = eval_acc_and_f1(answerable)

    decline = dict(answer="Not answerable", pred="Not answerable", score=1.0)
    acc_after, f1_after = eval_acc_and_f1(answerable + [decline])

    assert acc_before == pytest.approx(0.5)
    assert acc_after == pytest.approx(2 / 3)  # the decline counts in accuracy
    assert f1_after == pytest.approx(f1_before)  # ...and does nothing to F1


def test_wrong_answer_on_unanswerable_hurts_precision_only():
    base = [dict(answer="a", pred="a", score=1.0)]
    _, f1_base = eval_acc_and_f1(base)  # recall 1, precision 1
    wrong = dict(answer="Not answerable", pred="42", score=0.0)
    _, f1_wrong = eval_acc_and_f1(base + [wrong])  # precision 1/2, recall 1
    assert f1_base == pytest.approx(1.0)
    assert f1_wrong == pytest.approx(2 * 1.0 * 0.5 / 1.5)


# --------------------------------------------------------------------------
# accuracy / F1 arithmetic, worked by hand
# --------------------------------------------------------------------------
def test_acc_and_f1_hand_computed():
    samples = [
        # four answerable keys
        dict(answer="a", pred="a", score=1.0),
        dict(answer="b", pred="b", score=1.0),
        dict(answer="c", pred="c-ish", score=0.5),
        dict(answer="d", pred="Not answerable", score=0.0),  # missed
        # three unanswerable keys
        dict(answer="Not answerable", pred="Not answerable", score=1.0),  # good decline
        dict(answer="Not answerable", pred="x", score=0.0),  # hallucinated
        dict(answer="Not answerable", pred="y", score=0.0),  # hallucinated
        # a sample with no score is ignored
        dict(answer="e", pred="e"),
    ]
    # accuracy  = (1 + 1 + .5 + 0 + 1 + 0 + 0) / 7             = 3.5 / 7 = 0.5
    # recall    = (1 + 1 + .5 + 0) / 4 answerable keys           = 2.5 / 4 = 0.625
    # precision = 2.5 / 5 predictions != "Not answerable"
    #             (a, b, c-ish, x, y)                           = 0.5
    # F1        = 2 * .625 * .5 / (.625 + .5)                   = 0.5555...
    acc, f1 = eval_acc_and_f1(samples)
    assert acc == pytest.approx(0.5)
    assert f1 == pytest.approx(2 * 0.625 * 0.5 / 1.125)


def test_acc_and_f1_edge_cases():
    assert eval_acc_and_f1([]) == (0.0, 0.0)
    only_unanswerable = [dict(answer="Not answerable", pred="Not answerable", score=1.0)]
    acc, f1 = eval_acc_and_f1(only_unanswerable)
    assert acc == 1.0
    assert f1 == 0.0  # no answerable keys: recall undefined


# --------------------------------------------------------------------------
# security: no code execution on model output
# --------------------------------------------------------------------------
PAYLOAD = "[__import__('builtins').setattr(__import__('sys'), '_mmlb_pwned', True)]"


def test_literal_eval_rejects_malicious_list_payload():
    # eval(PAYLOAD) would run setattr(sys, ...). literal_eval refuses outright.
    with pytest.raises(ValueError):
        ast.literal_eval(PAYLOAD)


def test_scorer_does_not_execute_list_payload():
    assert not hasattr(sys, "_mmlb_pwned")
    items = _parse_list(PAYLOAD)
    assert isinstance(items, list)
    score = eval_score("['a']", PAYLOAD, "List")
    assert score == 0.0
    assert not hasattr(sys, "_mmlb_pwned")


def test_parse_list_forms():
    assert _parse_list("['a', 'b']") == ["a", "b"]
    assert _parse_list("[1, 2]") == [1, 2]
    assert _parse_list("[a, b]") == ["a", "b"]
    assert _parse_list("plain") == ["plain"]
    assert _parse_list(["x"]) == ["x"]


# --------------------------------------------------------------------------
# extraction step
# --------------------------------------------------------------------------
def test_parse_extraction():
    assert parse_extraction("Extracted answer: 18.29%\nAnswer format: Float") == "18.29%"
    assert parse_extraction("Extracted answer: ['a', 'b']\nAnswer format: List") == "['a', 'b']"
    assert parse_extraction("Extracted answer: Not answerable\nAnswer format: String") == "Not answerable"
    assert parse_extraction("I think it's 42") == FAILED_TO_EXTRACT
    assert parse_extraction("Failed") == FAILED_TO_EXTRACT
    assert parse_extraction("") == FAILED_TO_EXTRACT


class _StubClient:
    def __init__(self, reply=None, error=None):
        self.calls: list[dict] = []
        self._reply, self._error = reply, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        message = SimpleNamespace(content=self._reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_extract_answer_is_fully_parameterised():
    client = _StubClient("Extracted answer: 7\nAnswer format: Integer")
    out = extract_answer("How many?", "There are 7.", "PROMPT", client=client, model="any-model")
    assert out.startswith("Extracted answer: 7")
    call = client.calls[0]
    assert call["model"] == "any-model"
    assert call["temperature"] == 0.0
    assert len(call["messages"]) == 1 and call["messages"][0]["role"] == "user"
    content = call["messages"][0]["content"]
    assert "PROMPT" in content and "How many?" in content and "There are 7." in content


def test_extract_answer_returns_failed_on_error_or_empty():
    assert extract_answer("q", "o", "p", client=_StubClient(error=RuntimeError("x")), model="m") == "Failed"
    assert extract_answer("q", "o", "p", client=_StubClient(reply=None), model="m") == "Failed"


def test_bundled_prompt_loads():
    prompt = load_extraction_prompt()
    assert "Extracted answer:" in prompt and "Not answerable" in prompt


def test_score_response_end_to_end_with_stub_extractor():
    sample = dict(question="How many?", answer="7", answer_format="Int")
    scored = score_response(
        sample, "There are seven... 7 in total.",
        lambda q, r: "Extracted answer: 7\nAnswer format: Integer",
    )
    assert scored["pred"] == "7" and scored["score"] == 1.0

    failed = score_response(sample, "?", lambda q, r: "Failed")
    assert failed["pred"] == FAILED_TO_EXTRACT and failed["score"] == 0.0
