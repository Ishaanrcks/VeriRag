"""Memory preflight for the benchmark runner.

The gauge is always a mocked reader, never the real machine: these tests must give
the same answer on a 64 GB workstation and on the 8 GB laptop that motivated them.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from verirag.eval import mmlongbench_runner as R
from verirag.eval.mmlongbench_runner import GB, MemoryLow, MemoryStatus, StallDetected

# Real `vm_stat` and `sysctl vm.swapusage` output captured from the 8 GB development Mac.
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               66558.
Pages active:                            144482.
Pages inactive:                           94640.
Pages speculative:                        49045.
Pages throttled:                              0.
Pages wired down:                         93488.
Pages purgeable:                           5701.
"Translation faults":                1174459573.
File-backed pages:                       123091.
"""
SWAPUSAGE = "vm.swapusage: total = 3072.00M  used = 2225.88M  free = 846.12M  (encrypted)"
MEMINFO = "MemTotal:        8000000 kB\nMemAvailable:    3000000 kB\nSwapTotal:       2000000 kB\nSwapFree:         500000 kB\n"


def status(available_gb, total_gb=8.0, swap_used_gb=2.0, swap_total_gb=3.0):
    return MemoryStatus(int(available_gb * GB), int(total_gb * GB),
                        int(swap_used_gb * GB), int(swap_total_gb * GB))


# --------------------------------------------------------------------------
# reading the gauge
# --------------------------------------------------------------------------
def test_available_is_reclaimable_memory_not_the_near_zero_free_figure():
    available = R.parse_vm_stat(VM_STAT)
    # free + inactive + speculative + purgeable, in 16 KiB pages
    assert available == (66558 + 94640 + 49045 + 5701) * 16384
    assert available / GB == pytest.approx(3.29, abs=0.01)
    assert 66558 * 16384 / GB < 1.1  # "free" alone says ~1 GB: it understates what can be reclaimed


def test_active_and_wired_pages_are_not_counted_as_available():
    only_active = VM_STAT.replace("66558.", "0.").replace("94640.", "0.") \
        .replace("49045.", "0.").replace("5701.", "0.")
    assert R.parse_vm_stat(only_active) == 0


def test_parse_swapusage():
    used, total = R.parse_swapusage(SWAPUSAGE)
    assert total == 3072 * 1024**2 and used == int(2225.88 * 1024**2)


def test_parse_linux_meminfo():
    s = R.parse_meminfo(MEMINFO)
    assert (s.available_bytes, s.total_bytes) == (3000000 * 1024, 8000000 * 1024)
    assert s.swap_used_bytes == 1500000 * 1024


def test_unreadable_output_raises_a_value_error_so_the_reader_can_fail_open():
    with pytest.raises(ValueError):
        R.parse_vm_stat("garbage")
    with pytest.raises(ValueError):
        R.parse_swapusage("garbage")


def test_description_names_available_total_and_swap():
    assert status(1.2).describe() == "1.2 GB available of 8.0 GB; swap 2.0 of 3.0 GB used"


# --------------------------------------------------------------------------
# the check itself
# --------------------------------------------------------------------------
def test_below_the_floor_refuses_with_an_actionable_message():
    with pytest.raises(MemoryLow) as caught:
        R.check_memory(2.5, reader=lambda: status(1.1))
    message = str(caught.value)
    assert "Only 1.1 GB available of 8.0 GB" in message and "at least 2.5 GB" in message
    assert "start" in message and "Close other applications" in message and "swap" in message


def test_at_or_above_the_floor_passes_and_returns_the_reading():
    assert R.check_memory(2.5, reader=lambda: status(2.5)).available_gb == pytest.approx(2.5)
    assert R.check_memory(2.5, reader=lambda: status(7.9)) is not None


def test_just_below_the_floor_refuses():
    with pytest.raises(MemoryLow):
        R.check_memory(2.5, reader=lambda: status(2.49))


def test_an_unreadable_gauge_fails_open_and_zero_disables_the_check():
    assert R.check_memory(2.5, reader=lambda: None) is None
    assert R.check_memory(0, reader=lambda: status(0.0)) is not None


def test_dry_run_summary_says_whether_a_real_run_would_start():
    ok, would_start = R.memory_summary(2.5, reader=lambda: status(3.0))
    assert would_start and "OK to start" in ok and "3.0 GB available" in ok
    bad, would_start = R.memory_summary(2.5, reader=lambda: status(0.8))
    assert not would_start and "REFUSE" in bad and "0.8 GB available" in bad
    unknown, would_start = R.memory_summary(2.5, reader=lambda: None)
    assert would_start and "could not be read" in unknown


def test_defaults_are_ordered_and_justified():
    # starting needs more than staying; a healthy resident run read 0.8-1.5 GB available
    assert R.MIN_AVAILABLE_GB > R.MIN_AVAILABLE_GB_DURING
    assert R.MIN_AVAILABLE_GB_DURING < 0.8
    assert R.MIN_AVAILABLE_GB >= 2.5


# --------------------------------------------------------------------------
# inside a run
# --------------------------------------------------------------------------
def _q(i, doc="a.pdf"):
    return dict(id=i, doc_id=doc, doc_type="r", group="text", evidence_sources=["text"],
                evidence_pages=[1], cross_page=False, answer_format="Str", v2_status="unchanged",
                expected_verdict="answer", question=f"Question {i}?", ground_truth="x")


class Pipeline:
    def __init__(self, seconds=None):
        self.asked, self.seconds = [], seconds or {}

    def __call__(self, text):
        self.asked.append(text)
        return SimpleNamespace(abstained=True, answer=None, outcome="abstained_insufficient",
                               abstention_reason="stub", regeneration_count=0, citations=[],
                               gate1_verdict=None, gate2_verdict=None, pipeline_log=[],
                               verified_facts=[])


def _extract(question, text):
    return "Extracted answer: Not answerable\nAnswer format: String"


def _run(tmp_path, reader, pipeline=None, n=4, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    subset = tmp_path / "subset.json"
    subset.write_text(json.dumps({"name": "mem", "questions": [_q(i) for i in range(1, n + 1)]}))
    pipeline = pipeline or Pipeline()
    options = dict(out_dir=tmp_path / "out", ingested_docs=[{"source_path": "/x/a.pdf"}],
                   query_fn=pipeline, extract_fn=_extract, log=lambda m: None,
                   memory_reader=reader)
    options.update(kwargs)
    return R.run_benchmark(subset, tmp_path / "store", **options), pipeline


def test_low_memory_at_start_refuses_before_anything_runs(tmp_path):
    pipeline = Pipeline()
    with pytest.raises(MemoryLow, match="to start"):
        _run(tmp_path, lambda: status(1.0), pipeline=pipeline)
    assert pipeline.asked == []                      # no question was asked
    assert not (tmp_path / "out").exists()           # nothing was written, no checkpoint
    assert not (tmp_path / "store").exists()         # not even the store was touched


def test_start_and_during_use_different_floors(tmp_path):
    # 1.5 GB: too little to START (2.5) but fine to KEEP RUNNING (0.5)
    with pytest.raises(MemoryLow, match="to start"):
        _run(tmp_path, lambda: status(1.5))
    payload, _ = _run(tmp_path / "again", lambda: status(1.5), min_available_gb=1.0)
    assert len(payload["questions"]) == 4


def test_memory_squeezed_mid_run_stops_at_a_resumable_checkpoint(tmp_path):
    readings = iter([3.0, 3.0, 3.0, 0.2])  # start, before Q1, before Q2, then another app opens
    pipeline = Pipeline()
    with pytest.raises(MemoryLow, match="keep running") as caught:
        _run(tmp_path, lambda: status(next(readings)), pipeline=pipeline)
    assert pipeline.asked == ["Question 1?", "Question 2?"]       # stopped before Question 3
    assert "2 question(s) are checkpointed" in str(caught.value) and "resume" in str(caught.value)
    checkpoint = json.loads((tmp_path / "out" / "mem_PARTIAL.json").read_text())
    assert [r["id"] for r in checkpoint["questions"]] == [1, 2] and checkpoint["complete"] is False

    resumed = Pipeline()
    payload, _ = _run(tmp_path, lambda: status(3.0), pipeline=resumed)
    assert resumed.asked == ["Question 3?", "Question 4?"]         # only the rest ran
    assert sorted(r["id"] for r in payload["questions"]) == [1, 2, 3, 4]


def test_the_check_runs_before_every_question(tmp_path):
    calls = []

    def reader():
        calls.append(1)
        return status(3.0)

    _run(tmp_path, reader)
    assert len(calls) == 1 + 4  # once to start, once before each of the 4 questions


# --------------------------------------------------------------------------
# the slow-question guard (the check that caught the real stall)
# --------------------------------------------------------------------------
class SlowFirst(Pipeline):
    """Makes the recorded elapsed time of Q2 look like a paging machine."""


def test_a_wedged_question_stops_the_run_at_its_checkpoint(tmp_path, monkeypatch):
    real = R.process_question

    def slow_second(row, **kw):
        record = real(row, **kw)
        if row["id"] == 2:
            record["elapsed_s"] = 2008.0  # the question that really took 33 minutes
        return record

    monkeypatch.setattr(R, "process_question", slow_second)
    pipeline = Pipeline()
    with pytest.raises(StallDetected, match="Question 2 took 2008s") as caught:
        _run(tmp_path, lambda: status(3.0), pipeline=pipeline, max_seconds_per_question=900)
    assert pipeline.asked == ["Question 1?", "Question 2?"]
    assert "2 question(s) are checkpointed" in str(caught.value)


def test_a_slow_last_question_does_not_raise_and_zero_disables_the_guard(tmp_path, monkeypatch):
    real = R.process_question

    def always_slow(row, **kw):
        record = real(row, **kw)
        record["elapsed_s"] = 5000.0
        return record

    monkeypatch.setattr(R, "process_question", always_slow)
    payload, _ = _run(tmp_path, lambda: status(3.0), max_seconds_per_question=0)
    assert len(payload["questions"]) == 4
    # with the guard on, only the LAST question may be slow without stopping the run
    with pytest.raises(StallDetected):
        _run(tmp_path / "on", lambda: status(3.0), max_seconds_per_question=900)
