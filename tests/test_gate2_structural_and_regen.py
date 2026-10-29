"""Gate 2 structural-claim routing, single-chunk false-hedge probing, and the
regeneration feedback.

Built from the real questions that exposed the bugs: #6 (claims about section
headings scored by NLI at 0.0035-0.22 although the headings were in the text)
and #8 (a false hedge the regeneration loop could not resolve). The LLM calls
are stubbed; none of these tests loads a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import verirag.critic.claims as claims_mod
from verirag.critic import gate2_groundedness as g2
from verirag.critic.claims import _coerce_claim, structural_titles
from verirag.generation import generator as gen
from verirag.generation.generator import Citation, GeneratedAnswer
from verirag.graph import nodes
from verirag.retrieval.types import SearchHit

FIX = Path(__file__).parent / "fixtures"
Q6 = json.loads((FIX / "q6_structural_claims.json").read_text())
Q8 = json.loads((FIX / "q8_false_hedge.json").read_text())

HEADING_1 = ("Sharp divides in German and American views of security issues, "
             "from use of force to defense budgeting")


def _hit(chunk_id, text, content_type="text", page=1):
    return SearchHit(chunk_id=chunk_id, raw_text=text,
                     metadata={"content_type": content_type, "page": page})


def _answer(hits, text="answer"):
    citation = Citation(number=1, parent_text="", source_path="x.pdf", doc_id="d",
                        pages=[1], contributing_chunk_ids=[h.chunk_id for h in hits],
                        content_types=sorted({h.metadata["content_type"] for h in hits}))
    return GeneratedAnswer(answer_text=text, citations=[citation], format_used=None, hits_used=hits)


def _run_gate2(monkeypatch, answer, claim_dicts):
    payload = json.dumps(claim_dicts)
    monkeypatch.setattr(claims_mod, "chat_completion", lambda **kw: payload)
    return g2.check_groundedness(answer)


def _q6_hits():
    return [_hit(h["chunk_id"], h["text"], page=h["page"]) for h in Q6["hits"]]


# --------------------------------------------------------------------------
# BUG 1: routing
# --------------------------------------------------------------------------
def _coerced(text, label="factual"):
    return _coerce_claim({"text": text, "claim_type": label, "cited_numbers": [1]})


def test_q6_heading_claims_are_routed_as_structural():
    # the exact claims of question #6, iteration 1; the model called them "factual"
    for c in Q6["claims_by_iteration"][0]:
        coerced = _coerced(c["text"], c["type"])
        if '"' in c["text"] and "only chapter" not in c["text"] and c["type"] == "factual":
            assert coerced.claim_type == "structural", c["text"]
            assert coerced.titles


def test_structural_detection_ignores_passage_numbers_and_measurements():
    assert structural_titles('Passage 1 contains a section heading titled "Methods and Data".')
    # a number outside the quotes is a measurement, which presence cannot verify
    assert structural_titles('The "Results" section reports accuracy of 92% on the test set.') is None
    assert structural_titles("The paper states that the model has 8M parameters.") is None


def test_meta_and_absence_labels_are_not_overridden():
    only = Q6["claims_by_iteration"][0][0]  # "The only chapter or section heading ... is "X""
    assert only["type"] == "meta"
    assert _coerced(only["text"], "meta").claim_type == "meta"
    hedge = 'The passages do not state that the section "Americans and Germans differ in their views" discusses NATO.'
    assert _coerced(hedge, "factual").claim_type == "absence"


def test_passage_identity_claims_are_meta():
    c = _coerced("Passage 2 is an unrelated business management course document.")
    assert c.claim_type == "meta"


# --------------------------------------------------------------------------
# BUG 1: verification by presence
# --------------------------------------------------------------------------
def test_q6_answer_is_grounded_by_presence_not_nli(monkeypatch):
    claims = [dict(text=c["text"], claim_type=c["type"], cited_numbers=c["cited"])
              for c in Q6["claims_by_iteration"][0]]
    verdict = _run_gate2(monkeypatch, _answer(_q6_hits()), claims)
    labels = [v.verdict for v in verdict.claims]
    assert verdict.overall_grounded, labels
    assert labels.count("supported_presence") == 3
    assert "not_supported" not in labels


def test_missing_title_is_not_supported(monkeypatch):
    hits = [_hit("c1", "Some unrelated page text about budgets and defence.")]
    claim = dict(text=f'Passage 1 contains a section heading titled "{HEADING_1}".',
                 claim_type="factual", cited_numbers=[1])
    verdict = _run_gate2(monkeypatch, _answer(hits), [claim])
    assert [v.verdict for v in verdict.claims] == ["not_supported"]
    assert not verdict.overall_grounded


def test_title_matches_across_line_breaks_and_punctuation(monkeypatch):
    wrapped = "6\nPEW\nSharp divides in German and American views of\nsecurity issues, from use of force to\ndefense budgeting\nDifferences predominate."
    claim = dict(text=f'Passage 1 contains a section heading titled "{HEADING_1}".',
                 claim_type="factual", cited_numbers=[1])
    verdict = _run_gate2(monkeypatch, _answer([_hit("c1", wrapped)]), [claim])
    assert [v.verdict for v in verdict.claims] == ["supported_presence"]


def test_title_found_only_in_a_vision_caption_is_not_certified(monkeypatch):
    hits = [_hit("cap1", f"A chart headed {HEADING_1} with two bars.", content_type="figure_caption")]
    claim = dict(text=f'Passage 1 contains a section heading titled "{HEADING_1}".',
                 claim_type="factual", cited_numbers=[1])
    verdict = _run_gate2(monkeypatch, _answer(hits), [claim])
    assert [v.verdict for v in verdict.claims] == ["caption_only_uncertified"]


def test_section_content_must_also_be_present(monkeypatch):
    text = f"{HEADING_1}\nDifferences on security issues predominate when looking at public opinion."
    ok = dict(text=f'The section "{HEADING_1}" covers differences on security issues.',
              claim_type="factual", cited_numbers=[1])
    bad = dict(text=f'The section "{HEADING_1}" covers nuclear submarine procurement schedules.',
               claim_type="factual", cited_numbers=[1])
    verdict = _run_gate2(monkeypatch, _answer([_hit("c1", text)]), [ok, bad])
    assert [v.verdict for v in verdict.claims] == ["supported_presence", "not_supported"]


def test_stemming_matches_relation_and_relationship():
    evidence = ["The relationship between Germany and NATO is discussed."]
    assert g2._content_overlap("discusses Germany in relation to NATO", evidence) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# BUG 2: the false-hedge probe needs the words in ONE chunk
# --------------------------------------------------------------------------
HEDGE = "The provided documents do not contain any other chapter names addressing Germany's relationship with NATO."


def test_hedge_contradicted_by_one_chunk_is_a_false_hedge(monkeypatch):
    hits = [
        _hit("a", "Other chapter names addressing Germany's relationship with NATO are listed here."),
        _hit("b", "Budget tables for the fiscal year."),
    ]
    verdict = _run_gate2(monkeypatch, _answer(hits),
                         [dict(text=HEDGE, claim_type="absence", cited_numbers=[])])
    claim = verdict.claims[0]
    assert claim.verdict == "false_hedge"
    assert claim.evidence_chunk_ids == ["a"]  # the chunk that contradicts it


def test_hedge_whose_words_are_scattered_across_chunks_is_not_false(monkeypatch):
    # every word occurs SOMEWHERE in the context, but no one chunk covers the topic
    hits = [
        _hit("a", "Chapter names for the appendix."),
        _hit("b", "Germany and the other member states."),
        _hit("c", "The relationship with NATO addressing security."),
    ]
    verdict = _run_gate2(monkeypatch, _answer(hits),
                         [dict(text=HEDGE, claim_type="absence", cited_numbers=[])])
    assert verdict.claims[0].verdict == "absence_unverified"
    assert verdict.overall_grounded


# --------------------------------------------------------------------------
# BUG 2: the feedback the regeneration step sends
# --------------------------------------------------------------------------
def _q8_state():
    table, text = Q8["table"], Q8["text"]
    hits = [_hit(text["chunk_id"], text["text"], page=15),
            _hit(table["chunk_id"], table["text"], content_type="table", page=15)]
    topic = ("question of what percentage of West German respondents say the U.S. "
             "relationship is as important as the Russia relationship")
    unsupported = [
        g2.ClaimVerdict("The provided documents do not address the question of what percentage ...",
                        "absence", "false_hedge", evidence_chunk_ids=[table["chunk_id"]],
                        entailment_prob=1.0, probe_topic=topic),
        # the answer-bearing claim: every number is in the table, NLI could not follow it
        g2.ClaimVerdict('Among West Germans, 29% volunteered "Both" and 21% chose Russia.',
                        "factual", "not_supported", entailment_prob=0.49,
                        lexical_overlap=0.4, numbers_in_evidence=True),
        # a claim with a number the evidence does not contain
        g2.ClaimVerdict("Among West Germans, 77% chose the U.S.", "factual", "not_supported",
                        entailment_prob=0.01, lexical_overlap=0.5, numbers_in_evidence=False),
    ]
    verdict = g2.Gate2Verdict(overall_grounded=False, claims=unsupported, unsupported_claims=unsupported)
    return {"query": Q8["question"], "answer_format": "concise", "hits": hits,
            "gate2_verdict": verdict, "regeneration_count": 0, "llm_call_count": 1}, topic


def test_regeneration_feedback_distinguishes_the_three_failures(monkeypatch):
    captured = {}

    def fake_generate(query, hits, fmt, **kwargs):
        captured.update(kwargs)
        return GeneratedAnswer("a", [], fmt, hits)

    monkeypatch.setattr(nodes, "generate_answer", fake_generate)
    state, topic = _q8_state()
    nodes.regenerate_node(state)

    # the false hedge gets the passage text that contradicts it, not just a topic
    (hedge_topic, passage), = captured["false_hedges"]
    assert hedge_topic == topic
    assert "29%" in passage and "Both(VOL)" in passage
    assert captured["false_hedge_topics"] == []
    # the true-but-unverifiable claim is to be RESTATED, the invented one OMITTED
    assert captured["restate_claims"] == ['Among West Germans, 29% volunteered "Both" and 21% chose Russia.']
    assert captured["avoid_claims"] == ["Among West Germans, 77% chose the U.S."]


def test_false_hedge_without_a_locatable_passage_falls_back_to_the_topic(monkeypatch):
    captured = {}
    monkeypatch.setattr(nodes, "generate_answer",
                        lambda q, h, f, **kw: captured.update(kw) or GeneratedAnswer("a", [], f, h))
    state, topic = _q8_state()
    state["gate2_verdict"].unsupported_claims[0].evidence_chunk_ids = []
    nodes.regenerate_node(state)
    assert captured["false_hedges"] == [] and captured["false_hedge_topics"] == [topic]


def _capture_messages(monkeypatch):
    seen = {}

    def fake_chat(messages, **kwargs):
        seen["system"], seen["user"] = messages[0]["content"], messages[1]["content"]
        return "an answer [1]"

    monkeypatch.setattr(gen, "chat_completion", fake_chat)
    return seen


def test_passage_text_stays_out_of_the_system_prompt(monkeypatch):
    seen = _capture_messages(monkeypatch)
    snippet = "West | 43% | 29% | 21% UNIQUE-DOCUMENT-TEXT"
    gen.generate_answer("q?", [_hit("c", "context text")], gen.AnswerFormat.CONCISE,
                        false_hedges=[("the topic", snippet)],
                        restate_claims=["a claim"], avoid_claims=["another"])
    # injection-safety boundary: document text only in the user message
    assert "UNIQUE-DOCUMENT-TEXT" not in seen["system"]
    assert "UNIQUE-DOCUMENT-TEXT" in seen["user"]
    assert "the topic" in seen["system"]
    assert "Restate each one only if" in seen["system"]
    assert "Do not repeat them" in seen["system"]
    # the evidence instruction no longer lets the model simply keep the hedge
    assert "Maintain the hedge only if" not in seen["system"]
    assert "Keep a hedge only for a specific detail" in seen["system"]


def test_plain_topic_steer_is_unchanged_without_evidence(monkeypatch):
    seen = _capture_messages(monkeypatch)
    gen.generate_answer("q?", [_hit("c", "context text")], gen.AnswerFormat.CONCISE,
                        false_hedge_topics=["the topic"])
    assert "Maintain the hedge only if" in seen["system"]
    assert "Passage text bearing on your earlier hedge" not in seen["user"]


def test_best_passage_snippet():
    table = "| | US | Both(VOL) | Russia |\n| --- | --- | --- | --- |\n| West | 43% | 29% | 21% |"
    assert "Both(VOL)" in gen.best_passage_snippet("anything", table)  # a table is kept whole
    text = "\n".join(["Intro line about nothing."] * 30
                     + ["Nearly four-in-ten East Germans prefer close ties with Russia."]
                     + ["Outro line about nothing."] * 30)
    snippet = gen.best_passage_snippet("East Germans close ties with Russia", text, width=200)
    assert "East Germans" in snippet and len(snippet) < 400
    assert gen.best_passage_snippet("x", "") == ""
