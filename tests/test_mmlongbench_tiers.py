"""Tier selection for MMLongBench-Doc ingestion (pure functions, no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from verirag.eval import mmlongbench_runner as R
from verirag.eval.mmlongbench_tiers import coverage, doc_profiles, rank_documents


def q(i, doc, group, doc_type="Report", cross=False):
    return dict(id=i, doc_id=doc, group=group, doc_type=doc_type, cross_page=cross)


QUESTIONS = [
    q(1, "text_a.pdf", "text"), q(2, "text_b.pdf", "text"), q(3, "text_b.pdf", "text"),
    q(4, "mixed.pdf", "table"), q(5, "mixed.pdf", "chart"), q(6, "mixed.pdf", "unanswerable"),
    q(7, "slides.pdf", "figure", doc_type="Slides", cross=True),
    q(8, "huge.pdf", "layout", doc_type="Brochure"),
]
SIZES = {"text_a.pdf": 1_000_000, "text_b.pdf": 1_500_000, "mixed.pdf": 2_000_000,
         "slides.pdf": 3_000_000, "huge.pdf": 30_000_000}


def test_a_document_that_covers_more_comes_first_not_a_text_only_one():
    ranked = rank_documents(QUESTIONS, SIZES)
    assert ranked[0] == "mixed.pdf"            # table + chart + unanswerable
    assert ranked.index("slides.pdf") < ranked.index("text_a.pdf")   # figure, cross-page, new type
    assert ranked.index("text_b.pdf") < ranked.index("text_a.pdf")   # same labels, more questions


def test_oversized_files_are_excluded():
    assert "huge.pdf" not in rank_documents(QUESTIONS, SIZES)
    assert "huge.pdf" in rank_documents(QUESTIONS, SIZES, max_bytes=50_000_000)


def test_ranking_is_deterministic():
    assert rank_documents(QUESTIONS, SIZES) == rank_documents(list(reversed(QUESTIONS)), SIZES)


def test_tiers_nest_and_the_next_tier_adds_what_is_missing():
    tier1 = rank_documents(QUESTIONS, SIZES)[:1]
    tier2 = rank_documents(QUESTIONS, SIZES, already=tier1)
    assert tier1[0] not in tier2
    # tier 1 already covers table/chart/unanswerable, so tier 2 now prefers the slides
    assert tier2[0] == "slides.pdf"


def test_a_document_without_a_known_size_is_not_a_candidate():
    assert "mixed.pdf" not in rank_documents(QUESTIONS, {"text_a.pdf": 1})


def test_coverage_reports_groups_and_whether_every_group_is_present():
    cov = coverage(QUESTIONS, ["mixed.pdf", "slides.pdf"])
    assert cov["documents"] == 2 and cov["questions"] == 4
    assert cov["groups"] == {"chart": 1, "figure": 1, "table": 1, "unanswerable": 1}
    assert cov["cross_page_documents"] == 1 and cov["all_groups_present"] is False
    assert coverage(QUESTIONS, ["mixed.pdf", "slides.pdf", "text_a.pdf", "huge.pdf"])["all_groups_present"]


def test_profiles_count_questions_per_document():
    p = doc_profiles(QUESTIONS)
    assert p["text_b.pdf"].questions == 2 and p["slides.pdf"].cross_page and not p["mixed.pdf"].cross_page


def test_tag_keeps_tiers_from_overwriting_each_other(tmp_path):
    subset = tmp_path / "s.json"
    subset.write_text(json.dumps({"name": "bench", "questions": [dict(
        id=1, doc_id="a.pdf", doc_type="r", group="text", evidence_sources=["text"],
        evidence_pages=[1], cross_page=False, answer_format="Str", v2_status="unchanged",
        expected_verdict="answer", question="Q?", ground_truth="x")]}))
    fake = lambda text: SimpleNamespace(  # noqa: E731
        abstained=True, answer=None, outcome="abstained_insufficient", abstention_reason="s",
        regeneration_count=0, citations=[], gate1_verdict=None, gate2_verdict=None,
        pipeline_log=[], verified_facts=[])
    for tag in ("tier1", "tier2"):
        R.run_benchmark(subset, tmp_path / "store", out_dir=tmp_path / "out", tag=tag,
                        ingested_docs=[{"source_path": "/x/a.pdf"}], query_fn=fake,
                        extract_fn=lambda q, t: "Extracted answer: Not answerable\nAnswer format: String",
                        memory_reader=lambda: R.MemoryStatus(16 * R.GB, 32 * R.GB), log=lambda m: None)
    names = sorted(p.name for p in (tmp_path / "out").glob("*.json"))
    assert any("bench_tier1_" in n for n in names) and any("bench_tier2_" in n for n in names)
    assert "bench_tier1_PARTIAL.json" in names and "bench_tier2_PARTIAL.json" in names


# --------------------------------------------------------------------------
# --force: which documents the ingestion script skips
# --------------------------------------------------------------------------
LOG = [
    dict(doc_id="done.pdf", status="ingested"),
    dict(doc_id="failed.pdf", status="failed"),
    dict(doc_id="also_done.pdf", status="ingested"),
]


def test_without_force_documents_already_ingested_are_skipped():
    from verirag.eval.mmlongbench_tiers import already_ingested

    # a failed document is retried; only successes are skipped
    assert already_ingested(LOG) == {"done.pdf", "also_done.pdf"}
    assert already_ingested([]) == set()


def test_force_skips_nothing_so_every_document_is_re_ingested():
    from verirag.eval.mmlongbench_tiers import already_ingested

    assert already_ingested(LOG, force=True) == set()
