"""The full-benchmark runner: preflight, sequential run, strict/lenient report, resume.

No model and no store: the pipeline, the extractor and the document list are injected.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from verirag.eval import mmlongbench_runner as R
from verirag.generation.partial import VerifiedFact, render_partial_answer
from verirag.pipeline_log import LogEvent


def _q(i, doc, expected="answer", gt="29%", fmt="Float", group="table", cross=False, v2="unchanged"):
    return dict(id=i, doc_id=doc, doc_type="Research report", group=group,
                evidence_sources=["table"], evidence_pages=[3], cross_page=cross,
                answer_format=fmt, v2_status=v2, expected_verdict=expected,
                question=f"Question {i}?", ground_truth=gt)


QUESTIONS = [
    _q(1, "a.pdf"),                                                    # answered in full
    _q(2, "a.pdf", expected="decline", gt="Not answerable", fmt="None", group="unanswerable"),
    _q(3, "b.pdf"),                                                    # document NOT ingested
    _q(4, "a.pdf", v2="corrected"),                                    # partial; value only in the body
]
INGESTED = [{"source_path": "/data/a.pdf", "doc_id": "a-1"}]


def _write_subset(tmp_path, questions=QUESTIONS):
    path = tmp_path / "subset.json"
    path.write_text(json.dumps({"name": "unit_subset", "questions": questions}))
    return path


class FakePipeline:
    """Deterministic stand-in for run_query, keyed on the question text."""

    def __init__(self):
        self.asked: list[str] = []

    def __call__(self, text):
        self.asked.append(text)
        retrieval = LogEvent(stage="retrieval", summary="r", detail={"chunks": [
            {"doc_name": "a.pdf", "page": 3, "type": "table", "rerank_score": 0.93}]})
        base = dict(abstention_reason=None, regeneration_count=0, citations=[],
                    gate1_verdict=None, gate2_verdict=None, pipeline_log=[retrieval],
                    verified_facts=[])
        if "Question 1?" in text:
            return SimpleNamespace(abstained=False, answer="West Germans: 29% chose Both. [1]",
                                   outcome="verified", **base)
        if "Question 2?" in text:
            return SimpleNamespace(abstained=True, answer=None, outcome="abstained_insufficient",
                                   **dict(base, abstention_reason="not covered"))
        facts = [VerifiedFact(text="The note says Don't know responses are not shown.")]
        body = "The chart shows 29% volunteered Both. The passages do not say what that means."
        return SimpleNamespace(abstained=False, answer=render_partial_answer(body, facts),
                               outcome="partial", **dict(base, regeneration_count=2,
                                                         verified_facts=facts))


def _extract(question, text):
    for value in ("29%",):
        if value in text:
            return f"Extracted answer: {value}\nAnswer format: Float"
    return "Extracted answer: Not answerable\nAnswer format: String"


def healthy_memory():
    """A mocked gauge: these tests must not depend on the machine they run on."""
    return R.MemoryStatus(available_bytes=16 * R.GB, total_bytes=32 * R.GB)


def _run(tmp_path, pipeline=None, **kwargs):
    pipeline = pipeline or FakePipeline()
    options = dict(out_dir=tmp_path / "out", ingested_docs=INGESTED, query_fn=pipeline,
                   extract_fn=_extract, log=lambda m: None, only_ingested=True,
                   memory_reader=healthy_memory)
    options.update(kwargs)
    payload = R.run_benchmark(_write_subset(tmp_path), tmp_path / "store", **options)
    return payload, pipeline


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------
def test_refuses_to_start_when_a_needed_document_is_missing():
    with pytest.raises(R.RunnerError, match="not in the store"):
        R.make_plan(QUESTIONS, INGESTED)


def test_only_ingested_runs_just_the_questions_with_a_document():
    plan = R.make_plan(QUESTIONS, INGESTED, only_ingested=True)
    assert [q["id"] for q in plan.todo] == [1, 2, 4]
    assert [q["id"] for q in plan.skipped] == [3]
    assert plan.docs_missing == ["b.pdf"] and plan.docs_present == ["a.pdf"]


def test_allow_missing_records_them_as_skipped():
    plan = R.make_plan(QUESTIONS, INGESTED, allow_missing=True)
    assert [q["id"] for q in plan.skipped] == [3]


def test_ids_and_limit():
    assert [q["id"] for q in R.make_plan(QUESTIONS, INGESTED, only_ingested=True, ids=[2, 4]).todo] == [2, 4]
    assert len(R.make_plan(QUESTIONS, INGESTED, only_ingested=True, limit=1).todo) == 1
    # asking only for a question whose document IS present needs no allowance
    assert R.make_plan(QUESTIONS, INGESTED, ids=[1]).docs_missing == []


def test_scope_doc_names_the_file_so_the_resolver_can_filter():
    assert R.query_text(QUESTIONS[0], "pooled") == "Question 1?"
    assert R.query_text(QUESTIONS[0], "doc") == "a.pdf Question 1?"


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------
def test_partial_answer_scores_zero_strict_and_correct_lenient(tmp_path):
    payload, _ = _run(tmp_path)
    q4 = next(r for r in payload["questions"] if r["id"] == 4)
    assert q4["outcome"] == "partial" and q4["partial"] is True
    assert (q4["pred"], q4["score"]) == ("Not answerable", 0.0)               # STRICT
    assert (q4["lenient_pred"], q4["lenient_score"]) == ("29%", 1.0)          # LENIENT
    assert q4["verified_facts"] == ["The note says Don't know responses are not shown."]
    assert q4["retrieved"] == [dict(doc_name="a.pdf", page=3, type="table", rerank_score=0.93)]


def test_headline_is_strict_and_lenient_is_the_upper_bound(tmp_path):
    payload, _ = _run(tmp_path)
    strict, lenient = payload["metrics_strict"], payload["metrics_lenient"]
    assert strict["n"] == lenient["n"] == 3                  # the skipped question is excluded
    assert strict["accuracy"] == pytest.approx(2 / 3)        # Q1 and the correct decline
    assert lenient["accuracy"] == pytest.approx(1.0)         # Q4's unverified 29% credited
    assert lenient["accuracy"] > strict["accuracy"]
    assert [s["id"] for s in payload["skipped"]] == [3]


def test_outcome_metrics_treat_partial_as_its_own_outcome(tmp_path):
    outcome = _run(tmp_path)[0]["outcome_metrics"]
    assert (outcome["answered"], outcome["partial"], outcome["declined"]) == (1, 1, 1)
    assert outcome["partial_on_answerable"] == 1
    assert outcome["recall"] == pytest.approx(0.5) and outcome["recall_incl_partial"] == pytest.approx(1.0)


def test_breakdowns_slice_like_the_official_report(tmp_path):
    b = _run(tmp_path)[0]["breakdowns"]
    assert set(b["answerability"]) == {"single_page", "unanswerable"}
    assert b["v2_status"]["corrected"] == dict(n=1, strict_accuracy=0.0, lenient_accuracy=1.0)
    assert b["outcome"]["partial"]["n"] == 1 and "table" in b["evidence_source"]


def test_report_labels_strict_headline_and_lenient_upper_bound(tmp_path):
    payload, _ = _run(tmp_path)
    report = (tmp_path / "out" / f"unit_subset_{payload['date']}.md").read_text()
    assert "STRICT (headline)" in report
    assert "LENIENT (upper bound, includes unverified statements)" in report
    assert report.index("STRICT") < report.index("LENIENT")
    assert "partial (some claims unverified)" in report and "1 skipped" in report


def test_scope_doc_prefixes_every_question_with_its_file(tmp_path):
    _, pipeline = _run(tmp_path, scope="doc")
    assert pipeline.asked == ["a.pdf Question 1?", "a.pdf Question 2?", "a.pdf Question 4?"]


# --------------------------------------------------------------------------
# sequential, checkpointed, resumable
# --------------------------------------------------------------------------
def test_questions_run_one_at_a_time_in_order_and_checkpoint_each(tmp_path):
    seen_checkpoints = []

    def spying_pipeline(text, _inner=FakePipeline()):
        checkpoint = tmp_path / "out" / "unit_subset_PARTIAL.json"
        seen_checkpoints.append(len(json.loads(checkpoint.read_text())["questions"]) if checkpoint.exists() else 0)
        return _inner(text)

    _run(tmp_path, pipeline=spying_pipeline)
    assert seen_checkpoints == [0, 1, 2]  # written after every question, before the next starts


def test_resume_skips_what_is_done_and_keeps_its_results(tmp_path):
    first = FakePipeline()
    _run(tmp_path, pipeline=first, ids=[1, 2])               # 'interrupted' after two questions
    assert first.asked == ["Question 1?", "Question 2?"]
    second = FakePipeline()
    payload, _ = _run(tmp_path, pipeline=second)             # the full run resumes
    assert second.asked == ["Question 4?"]                   # only the remaining one ran
    assert sorted(r["id"] for r in payload["questions"]) == [1, 2, 4]


def test_no_resume_starts_over(tmp_path):
    _run(tmp_path, ids=[1, 2])
    again = FakePipeline()
    _run(tmp_path, pipeline=again, resume=False)
    assert len(again.asked) == 3
