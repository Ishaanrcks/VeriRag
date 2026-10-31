"""``partial`` as a first-class harness outcome (Queensland Q44 and Q58).

Both were scored as "answered" because the harness only had refusal-phrase
detection to tell a decline from an answer. Partial-answer mode is a real system
behaviour, so the harness now takes the pipeline's own ``outcome`` signal.
No model: ``run_query`` is stubbed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from verirag.eval import harness as H
from verirag.generation.partial import UNVERIFIED_NOTE, VERIFIED_FACTS_HEADING, VerifiedFact

# Q44: an aggregation question the system cannot census; the "Three papers" count is
# in the body, unverified. The opening is NOT a refusal phrase.
Q44_TEXT = (
    "Three papers in the corpus apply graph neural networks. ExpressGNN uses a GNN "
    "variant for variational inference in Markov Logic Networks [1].\n\n"
    f"{VERIFIED_FACTS_HEADING}\n"
    "- ExpressGNN uses a GNN variant for variational inference in Markov Logic Networks. "
    "[1] (EFFICIENT PROBABILISTIC LOGIC REASONING.pdf, p. 11)\n\n"
    f"{UNVERIFIED_NOTE}"
)
# Q58: a false-premise question; the opening CORRECTS the premise rather than refusing.
Q58_TEXT = (
    "The dual-head design appears in the QpiGNN paper, not in GFM-RAG: QpiGNN employs "
    "a dual-head architecture [2].\n\n"
    f"{VERIFIED_FACTS_HEADING}\n"
    "- The QpiGNN paper employs a dual-head architecture. [2] (2605.04847v2.pdf, p. 4)\n\n"
    f"{UNVERIFIED_NOTE}"
)
# A Q54-style partial whose opening IS a refusal phrase ("do not give").
Q54_PARTIAL = (
    "The passages do not give a numeric learning rate for training QpiGNN. "
    "Algorithm 1 lists a learning rate as an input [1].\n\n"
    f"{VERIFIED_FACTS_HEADING}\n- Algorithm 1 lists a learning rate as an input. [1]\n\n{UNVERIFIED_NOTE}"
)


# --------------------------------------------------------------------------
# classification: the pipeline's signal, not the wording
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text", [Q44_TEXT, Q58_TEXT])
def test_q44_and_q58_classify_as_partial(text):
    assert H.classify_outcome(False, text, "partial") == "partial"


@pytest.mark.parametrize("text", [Q44_TEXT, Q58_TEXT])
def test_without_the_pipeline_signal_they_were_forced_into_answered(text):
    # what the harness did before: no refusal phrase opening -> "answered"
    assert H.classify_outcome(False, text) == "answered"
    assert H.classify_outcome(False, text, "verified") == "answered"


def test_a_partial_with_a_refusal_phrase_opening_is_still_partial():
    assert H.classify_outcome(False, Q54_PARTIAL) == "refused"  # text alone would say refused
    assert H.classify_outcome(False, Q54_PARTIAL, "partial") == "partial"


def test_the_other_outcomes_are_unchanged():
    assert H.classify_outcome(True, None, "abstained_ungrounded") == "abstained"
    assert H.classify_outcome(True, None, "partial") == "abstained"  # abstention wins
    assert H.classify_outcome(False, "The provided documents do not address this question.", "verified") == "refused"
    assert H.classify_outcome(False, "GFM-RAG has 8M parameters [1].", "verified") == "answered"


# --------------------------------------------------------------------------
# metrics: where partial sits
# --------------------------------------------------------------------------
def _rec(i, expected, outcome, correct=None):
    return dict(id=i, group="g", expected_verdict=expected, outcome=outcome,
                correct=correct, regeneration_count=0)


def test_partial_is_counted_in_fpr_but_not_in_recall_or_hallucinations():
    # the Queensland decline-expected shape: Q44 and Q58 partial, one verified answer, rest declined
    records = [_rec(44, "decline", "partial"), _rec(58, "decline", "partial"),
               _rec(51, "decline", "answered"), _rec(49, "decline", "abstained"),
               _rec(50, "decline", "refused"), _rec(1, "answer", "answered", True),
               _rec(2, "answer", "partial", False)]
    m = H.compute_metrics(records)
    assert (m.partial, m.partial_on_decline, m.partial_on_answerable) == (3, 2, 1)
    assert m.false_positive_rate == pytest.approx(3 / 5)               # any answer
    assert m.false_positive_rate_verified_only == pytest.approx(1 / 5)  # partial excluded
    assert m.recall == pytest.approx(1 / 2) and m.recall_incl_partial == pytest.approx(1.0)
    assert m.refusal_precision == pytest.approx(1.0)                    # partial is not a decline
    assert m.hallucination_rate == 0.0 and m.partial_wrong == [2]
    assert m.by_group["g"]["partial"] == 3 and m.by_group["g"]["declined"] == 2


# --------------------------------------------------------------------------
# end to end through run_question_set (run_query stubbed)
# --------------------------------------------------------------------------
def _result(text, outcome, facts=()):
    return SimpleNamespace(
        abstained=False, answer=text, abstention_reason=None, regeneration_count=2,
        citations=[], trace=[], outcome=outcome,
        verified_facts=[VerifiedFact(text=f) for f in facts],
    )


def _run(monkeypatch, tmp_path, questions, results):
    import verirag.graph.build as build
    import verirag.retrieval.vector_store as vs

    queue = list(results)
    monkeypatch.setattr(build, "run_query", lambda *a, **k: queue.pop(0))
    monkeypatch.setattr(vs, "list_ingested_documents", lambda *a, **k: [])
    return H.run_question_set({"name": "t", "questions": questions}, tmp_path / "store")


QUESTIONS = [
    dict(id=44, group="aggregation", question="Which papers apply GNNs?",
         ground_truth="n/a", expected_verdict="decline"),
    dict(id=58, group="false_premise", question="How does the dual-head help GFM-RAG?",
         ground_truth="false premise", expected_verdict="decline"),
]


def test_q44_and_q58_are_reclassified_end_to_end(monkeypatch, tmp_path):
    payload = _run(monkeypatch, tmp_path, QUESTIONS, [
        _result(Q44_TEXT, "partial", ["ExpressGNN uses a GNN variant ..."]),
        _result(Q58_TEXT, "partial", ["The QpiGNN paper employs a dual-head architecture."]),
    ])
    records = {r["id"]: r for r in payload["questions"]}
    assert records[44]["outcome"] == records[58]["outcome"] == "partial"
    assert records[44]["pipeline_outcome"] == "partial"
    assert records[44]["verified_facts"] == ["ExpressGNN uses a GNN variant ..."]
    assert not records[44]["meets_expectation"]  # an answer was returned where a decline was expected
    metrics = payload["metrics"]
    assert (metrics["partial"], metrics["partial_on_decline"]) == (2, 2)
    assert metrics["false_positive_rate"] == 1.0 and metrics["false_positive_rate_verified_only"] == 0.0


def test_partial_correctness_is_checked_against_the_verified_facts_only(monkeypatch, tmp_path):
    # "Three papers" is in the body but NOT in the Verified facts
    question = dict(QUESTIONS[0], expected_verdict="answer", must_contain=[["three papers"]])
    payload = _run(monkeypatch, tmp_path, [question], [_result(Q44_TEXT, "partial")])
    record = payload["questions"][0]
    assert record["correct"] is False and record["correct_lenient"] is True
    assert payload["metrics"]["partial_wrong"] == [44]
    assert payload["metrics"]["hallucination_rate"] == 0.0


def test_report_and_baseline_diff_know_about_partial(monkeypatch, tmp_path):
    payload = _run(monkeypatch, tmp_path, QUESTIONS, [
        _result(Q44_TEXT, "partial"), _result(Q58_TEXT, "partial")])
    baseline = {"name": "old", "date": "d", "questions": [
        {"id": 44, "outcome": "abstained"}, {"id": 58, "outcome": "abstained"}]}
    report = H.render_markdown(payload, baseline)
    assert "partial (some claims unverified)" in report
    assert "false positive rate, verified answers only" in report
    assert "accepted partial-mode tradeoff" in report
    assert "now partial" in report
    assert "wanted decline, got partial" in report
