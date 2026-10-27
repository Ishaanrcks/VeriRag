"""Offline tests for the Pipeline Log helpers and node-level event emission.

No models, no network: hits are plain namespaces, and the one node test uses
scores high enough that Gate 1 decides by threshold without an LLM call.
"""

from __future__ import annotations

from types import SimpleNamespace

from verirag.pipeline_log import (
    LogEvent,
    abstain_parts,
    claim_verdict_bucket,
    context_event,
    doc_name,
    gate1_event,
    gate1_line,
    gate2_event,
    gate2_rows,
    included_lines,
    new_log,
    retrieval_event,
    retrieval_rows,
    truncate,
    type_label,
)


def _hit(chunk_id, text, *, path="/a/b/paper.pdf", page=3, ctype="text",
         rerank=0.9, fused=0.03, **extra):
    metadata = {"source_path": path, "page": page, "content_type": ctype, **extra}
    return SimpleNamespace(chunk_id=chunk_id, raw_text=text, metadata=metadata,
                           rerank_score=rerank, fused_score=fused)


def test_new_log_and_event_defaults():
    assert new_log() == []
    event = LogEvent(stage="answer", summary="Answered")
    assert event.detail == {} and event.timestamp > 0


def test_truncate_collapses_whitespace_and_marks_cut():
    assert truncate("a  b\n c", 50) == "a b c"
    out = truncate("x" * 300, 100)
    assert len(out) == 100 and out.endswith("…")


def test_doc_name_is_never_a_path():
    assert doc_name({"source_path": "/Users/me/data/raw/paper.pdf"}) == "paper.pdf"
    assert doc_name({"source_path": "https://example.org/a/b",
                     "source_type": "external"}) == "example.org"
    assert doc_name({}) == "unknown"


def test_type_label_maps_figure_caption():
    assert type_label("table") == "table"
    assert type_label("figure_caption") == "image_caption"
    assert type_label("text") == "text" and type_label(None) == "text"


def test_retrieval_event_counts_and_stores_200_char_snippet():
    hits = [_hit("c1", "word " * 200), _hit("c2", "short", path="/x/other.pdf")]
    event = retrieval_event(hits)
    assert event.stage == "retrieval"
    assert event.detail["chunk_count"] == 2 and event.detail["doc_count"] == 2
    assert event.summary == "Retrieved 2 chunks from 2 documents"
    first = event.detail["chunks"][0]
    assert first["rank"] == 1 and first["doc_name"] == "paper.pdf"
    assert len(first["snippet"]) <= 200 and first["snippet"].endswith("…")
    assert first["score"] == 0.9 and first["rrf_score"] == 0.03


def test_single_document_is_not_pluralised():
    assert "1 document" in retrieval_event([_hit("c1", "t")]).summary
    assert "1 chunk " in retrieval_event([_hit("c1", "t")]).summary


def test_retrieval_rows_use_filename_and_100_char_snippet():
    event = retrieval_event([_hit("c1", "y" * 500, ctype="figure_caption", page=None)])
    row = retrieval_rows(event.detail["chunks"])[0]
    assert row["Document"] == "paper.pdf" and "/" not in row["Document"]
    assert row["Type"] == "image caption" and row["Page"] == "-"
    assert len(row["Snippet"]) <= 100 and row["Snippet"].endswith("…")
    assert set(row) == {"Rank", "Document", "Page", "Type", "Score", "Snippet"}


def test_context_event_includes_only_chunks_present_in_context():
    present, absent = _hit("c1", "alpha passage"), _hit("c2", "beta passage")
    context = "[1] (source: paper.pdf)\nparent text ... alpha passage ..."
    event = context_event([present, absent], context)
    assert event.detail["full_context"] == context  # unmodified
    assert event.detail["char_count"] == len(context)
    assert [c["rank"] for c in event.detail["chunks_included"]] == [1]
    assert included_lines(event.detail["chunks_included"]) == [
        "paper.pdf - page 3 - text"
    ]


def test_gate1_event_and_line_variants():
    common = dict(threshold_high=0.5, threshold_low=0.05, llm_reason=None)
    high = gate1_event(score=0.8, verdict="sufficient",
                       decision_path="threshold_high", **common)
    assert high.detail["verdict"] == "pass" and high.detail["threshold"] == 0.5
    assert gate1_line(high.detail).startswith("PASS (score: 0.80 / threshold 0.50)")

    low = gate1_event(score=0.02, verdict="insufficient",
                      decision_path="threshold_low", **common)
    assert low.detail["verdict"] == "fail" and low.detail["threshold"] == 0.05
    assert gate1_line(low.detail).startswith("FAIL (score: 0.02 / threshold 0.05)")

    judged = gate1_event(score=0.3, verdict="sufficient",
                         decision_path="llm_escalation",
                         threshold_high=0.5, threshold_low=0.05,
                         llm_reason="Passages state the answer.")
    assert "LLM judge decided" in gate1_line(judged.detail)
    assert "0.05-0.50" in gate1_line(judged.detail)


def test_claim_verdict_buckets():
    assert claim_verdict_bucket("supported_single") == "grounded"
    assert claim_verdict_bucket("supported_presence") == "grounded"
    for blocking in ("not_supported", "false_hedge", "caption_only_uncertified",
                     "external_unsupported"):
        assert claim_verdict_bucket(blocking) == "ungrounded"
    # Non-blocking, unscored claims are neither grounded nor ungrounded.
    assert claim_verdict_bucket("meta_excluded") == "excluded"
    assert claim_verdict_bucket("absence_unverified") == "excluded"


def test_gate2_event_maps_source_types_and_rows_truncate_claims():
    cv = lambda text, verdict, prob, ids: SimpleNamespace(  # noqa: E731
        claim_text=text, verdict=verdict, entailment_prob=prob,
        evidence_chunk_ids=ids)
    claims = [
        cv("A table-backed claim " + "z" * 200, "supported_presence", 0.91, ["t1", "p1"]),
        cv("An unsupported claim", "not_supported", 0.12, ["p1"]),
        cv("About the passages", "meta_excluded", None, []),
    ]
    event = gate2_event(claims, {"t1": "table", "p1": "text"})
    assert event.summary == "Gate 2 checked 3 claims, 1 ungrounded"
    first = event.detail["claims"][0]
    assert first["source_type"] == "table + text" and first["verdict"] == "grounded"
    assert event.detail["claims"][2]["source_type"] is None

    rows = gate2_rows(event.detail["claims"])
    assert len(rows[0]["Claim"]) <= 80 and rows[0]["Claim"].endswith("…")
    assert [r["Verdict"] for r in rows] == ["Grounded", "Ungrounded", "Excluded"]
    assert rows[2]["Source Type"] == "-"
    # Headed "Score", not "NLI Score": presence/absence verdicts carry a
    # token-overlap fraction in this column.
    assert set(rows[0]) == {"Claim", "Source Type", "Score", "Verdict"}
    # not_supported has no supporting chunks, so it has no Source Type.
    assert rows[1]["Source Type"] == "-"


def test_gate2_source_type_comes_from_supporting_chunks_only():
    cv = lambda text, verdict, prob, ids: SimpleNamespace(  # noqa: E731
        claim_text=text, verdict=verdict, entailment_prob=prob,
        evidence_chunk_ids=ids)
    types = {"t1": "text", "t2": "text", "f1": "figure_caption"}
    claims = [
        # Support set = the two text chunks (the caption did not decide it).
        cv("combined", "supported_combination", 0.99, ["t1", "t2"]),
        cv("caption only", "caption_only_uncertified", 0.9, ["f1"]),
        # Chunks were checked but none supported the claim.
        cv("unsupported", "not_supported", 0.0, ["t1", "f1"]),
        cv("hedge", "false_hedge", 1.0, []),
    ]
    out = gate2_event(claims, types).detail["claims"]
    assert [c["source_type"] for c in out] == ["text", "image_caption", None, None]


def test_abstain_parts_splits_headline_and_truncates_claim_bullets():
    reason = ("Could not produce a fully grounded answer after 2 regeneration "
              "attempt(s). These claims could not be verified against the "
              "sources:\n- " + "c" * 400 + "\n- short claim")
    headline, bullets = abstain_parts(reason)
    assert headline.startswith("Could not produce")
    assert len(bullets) == 2 and len(bullets[0]) <= 160
    assert bullets[0].endswith("…") and bullets[1] == "short claim"
    assert abstain_parts("") == ("No reason recorded.", [])


def test_gate1_node_emits_event_without_changing_its_verdict():
    # Imported here: the graph package pulls in the critic/retrieval stack.
    from verirag.graph.nodes import gate1_node

    state = {"query": "q", "hits": [_hit("c1", "text", rerank=0.9)],
             "llm_call_count": 0}
    update = gate1_node(state)
    assert update["gate1_verdict"].verdict == "sufficient"  # unchanged behaviour
    assert update["llm_call_count"] == 0  # threshold path: no LLM call
    (event,) = update["pipeline_log"]
    assert event.stage == "gate1" and event.detail["verdict"] == "pass"
    assert event.detail["score"] == 0.9
