"""Partial-answer mode: answer when Gate 2 verified SOME claims, decline when it
verified none, and say exactly which claims passed.

No model and no network: Gate 2's verdicts are built by hand.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from verirag.critic.gate2_groundedness import ClaimVerdict, Gate2Verdict
from verirag.eval.mmlongbench_scorer import extract_answer
from verirag.generation import generator as gen
from verirag.generation.generator import Citation, GeneratedAnswer
from verirag.generation.partial import (
    UNVERIFIED_NOTE,
    VERIFIED_FACTS_HEADING,
    collect_verified_facts,
    render_partial_answer,
    split_verified_facts,
)
from verirag.graph import build, nodes
from verirag.observability import metrics
from verirag.retrieval.types import SearchHit

VERIFIED_CLAIM = 'A chart lists West Germany at 29% for "Both (VOL)".'
SECOND_VERIFIED = "A chart lists West Germany at 21% for Russia."
UNVERIFIED_CLAIM = "West Germans therefore view both relationships as equally important."
ANSWER = (
    f"{VERIFIED_CLAIM} {SECOND_VERIFIED} [1] "
    f"{UNVERIFIED_CLAIM}"
)


def _verdict(text, label, chunk_ids=("t1",)):
    return ClaimVerdict(text, "factual", label, evidence_chunk_ids=list(chunk_ids))


def _gate2(*claims):
    claims = list(claims)
    unsupported = [c for c in claims if c.verdict in
                   ("not_supported", "false_hedge", "external_unsupported", "caption_only_uncertified")]
    return Gate2Verdict(overall_grounded=not unsupported, claims=claims, unsupported_claims=unsupported)


def _generated():
    hit = SearchHit(chunk_id="t1", raw_text="| West | 29% |", metadata={"content_type": "table", "page": 15})
    citation = Citation(number=1, parent_text="", source_path="/data/PG_2020.03.09_US-Germany.pdf",
                        doc_id="d", pages=[15], contributing_chunk_ids=["t1"], content_types=["table"])
    return GeneratedAnswer(answer_text=ANSWER, citations=[citation], format_used=None, hits_used=[hit])


def _state(*claims, regens=2, summarization=False):
    return {"gate2_verdict": _gate2(*claims), "generated_answer": _generated(),
            "regeneration_count": regens, "llm_call_count": 3, "is_summarization_intent": summarization}


# --------------------------------------------------------------------------
# the decision rule
# --------------------------------------------------------------------------
def test_all_claims_verified_is_unchanged():
    assert build._route_after_gate2(_state(_verdict("a", "supported_single"))) == "finalize_verified"


def test_some_verified_after_the_last_regeneration_answers_partially():
    state = _state(_verdict(VERIFIED_CLAIM, "supported_presence"), _verdict(UNVERIFIED_CLAIM, "not_supported"))
    assert build._route_after_gate2(state) == "finalize_partial"


def test_zero_verified_claims_still_declines():
    state = _state(_verdict(UNVERIFIED_CLAIM, "not_supported"), _verdict("h", "false_hedge"),
                   _verdict("m", "meta_excluded"), _verdict("a", "absence_unverified"))
    assert build._route_after_gate2(state) == "abstain_ungrounded"


def test_regeneration_still_comes_first_while_attempts_remain():
    state = _state(_verdict(VERIFIED_CLAIM, "supported_single"), _verdict(UNVERIFIED_CLAIM, "not_supported"), regens=0)
    assert build._route_after_gate2(state) == "regenerate"


def test_summarization_path_follows_the_same_rule():
    some = _state(_verdict("a", "supported_single"), _verdict("b", "not_supported"), summarization=True, regens=0)
    none = _state(_verdict("b", "not_supported"), summarization=True, regens=0)
    assert build._route_after_gate2(some) == "finalize_partial"
    assert build._route_after_gate2(none) == "abstain_ungrounded"


def test_gate1_insufficient_coverage_is_unchanged():
    state = {"gate1_verdict": SimpleNamespace(verdict="insufficient"), "corpus_only": True}
    assert build._route_after_gate1(state) == "abstain_insufficient_coverage"


@pytest.mark.parametrize("label", ["meta_excluded", "absence_unverified", "false_hedge",
                                   "caption_only_uncertified", "external_unsupported", "not_supported"])
def test_only_checked_claims_count_as_verified(label):
    assert not nodes.has_verified_claim(_state(_verdict("x", label)))


@pytest.mark.parametrize("label", ["supported_single", "supported_combination", "supported_presence"])
def test_each_supported_label_counts_as_verified(label):
    assert nodes.has_verified_claim(_state(_verdict("x", label)))


# --------------------------------------------------------------------------
# what a partial answer contains
# --------------------------------------------------------------------------
def _finalize():
    state = _state(
        _verdict(VERIFIED_CLAIM, "supported_presence"),
        _verdict(SECOND_VERIFIED, "supported_single"),
        _verdict(UNVERIFIED_CLAIM, "not_supported"),
        _verdict("The passages do not say more.", "absence_unverified"),
        _verdict("A comment about the passages.", "meta_excluded"),
    )
    return nodes.finalize_partial_node(state), state


def test_partial_answer_returns_the_text_unchanged_plus_a_verified_list():
    update, _ = _finalize()
    text = update["final_answer"]
    assert update["abstained"] is False and update["answer_is_partial"] is True
    assert text.startswith(ANSWER)  # nothing removed or reworded
    assert VERIFIED_FACTS_HEADING in text and UNVERIFIED_NOTE in text


def test_verified_list_has_exactly_the_verified_claims_with_sources():
    update, _ = _finalize()
    body, facts = split_verified_facts(update["final_answer"])
    lines = [ln for ln in facts.splitlines() if ln.startswith("- ")]
    assert len(lines) == 2
    assert f"- {VERIFIED_CLAIM} [1] (PG_2020.03.09_US-Germany.pdf, p. 15)" in lines
    assert any(SECOND_VERIFIED in ln for ln in lines)
    # the claim that was NOT verified stays visible in the answer, not in the list
    assert UNVERIFIED_CLAIM in body and UNVERIFIED_CLAIM not in facts
    assert "do not say more" not in facts and "comment about" not in facts
    assert [f.text for f in update["verified_facts"]] == [VERIFIED_CLAIM, SECOND_VERIFIED]


def test_facts_are_deduplicated_and_cite_their_source_chunk():
    verdicts = [_verdict(VERIFIED_CLAIM, "supported_single"),
                _verdict("  " + VERIFIED_CLAIM.upper() + " ", "supported_presence")]
    answer = _generated()
    facts = collect_verified_facts(verdicts, answer.citations, answer.hits_used)
    assert len(facts) == 1
    assert (facts[0].citation_numbers, facts[0].pages, facts[0].chunk_ids) == ([1], [15], ["t1"])


def test_outcome_is_partial_and_not_counted_as_an_abstention():
    assert build._derive_outcome({"abstained": False, "answer_is_partial": True}) == "partial"
    assert build._derive_outcome({"abstained": False}) == "verified"
    assert build._derive_outcome({"abstained": True, "gate2_verdict": _gate2(_verdict("x", "not_supported"))}) \
        == "abstained_ungrounded"


def test_zero_verified_decline_is_unchanged():
    state = _state(_verdict(UNVERIFIED_CLAIM, "not_supported"))
    update = nodes.abstain_ungrounded_node(state)
    assert update["abstained"] is True and update["final_answer"] is None
    assert UNVERIFIED_CLAIM in update["abstention_reason"]


def test_render_and_split_round_trip():
    facts = collect_verified_facts([_verdict(VERIFIED_CLAIM, "supported_single")],
                                   _generated().citations, _generated().hits_used)
    out = render_partial_answer("Body text. [1]", facts)
    body, section = split_verified_facts(out)
    assert body == "Body text. [1]" and section.startswith("- ") and UNVERIFIED_NOTE not in section
    assert split_verified_facts("Just an answer.") == ("Just an answer.", None)


def test_metrics_log_and_summary_accept_partial(tmp_path):
    db = str(tmp_path / "m.db")
    base = dict(timestamp="t", query="q", answer_format="concise", corpus_only=True,
                regeneration_count=2, external_fetch_attempted=False, llm_call_count=5,
                total_latency_ms=1.0, node_timings=[("gate2", 1.0)])
    metrics.log_query_metrics(metrics.QueryMetricsRecord(outcome="partial", **base), db)
    metrics.log_query_metrics(metrics.QueryMetricsRecord(outcome="abstained_ungrounded", **base), db)
    summary = metrics.summarize_metrics(db)
    assert summary.total_queries == 2 and summary.abstention_rate == pytest.approx(0.5)


# --------------------------------------------------------------------------
# the scorer reads the verified facts
# --------------------------------------------------------------------------
class _Client:
    def __init__(self):
        self.messages = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.messages = kw["messages"][0]["content"]
        msg = SimpleNamespace(content="Extracted answer: 29%\nAnswer format: Float")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


PARTIAL_OUTPUT = render_partial_answer(
    "The provided documents do not address this question as worded. " + ANSWER,
    collect_verified_facts([_verdict(VERIFIED_CLAIM, "supported_presence")],
                           _generated().citations, _generated().hits_used),
)


def test_extractor_is_pointed_at_the_verified_facts_section():
    client = _Client()
    extract_answer("q?", PARTIAL_OUTPUT, "PROMPT", client=client, model="m")
    sent = client.messages
    assert "ends with a 'Verified facts' list" in sent
    assert "even when the opening sentences hedge or decline" in sent
    tail = sent.split("Verified facts:\n", 1)[1]
    assert VERIFIED_CLAIM in tail and UNVERIFIED_CLAIM not in tail


def test_extractor_request_is_unchanged_without_a_verified_facts_section():
    client = _Client()
    extract_answer("q?", "A plain verified answer. [1]", "PROMPT", client=client, model="m")
    assert "Verified facts" not in client.messages


# --------------------------------------------------------------------------
# the prompt no longer demands a refusal opening when something was found
# --------------------------------------------------------------------------
def test_system_prompt_separates_nothing_found_from_something_found():
    prompt = " ".join(gen._SYSTEM_PROMPT.split())  # the prompt is hard-wrapped
    assert "If the passages contain NOTHING relevant" in prompt
    assert "LEAD with what they do" in prompt
    assert "Do not open with a refusal" in prompt
    # the old unconditional instruction is gone, but a true refusal is kept
    assert "If the passages do not actually address the question" not in prompt
    assert '"The provided documents do not address this question."' in prompt


# --------------------------------------------------------------------------
# the generator must commit to the value, not describe the passages
# --------------------------------------------------------------------------
def test_system_prompt_requires_leading_with_the_direct_answer():
    prompt = " ".join(gen._SYSTEM_PROMPT.split())
    assert "LEAD WITH THE DIRECT ANSWER" in prompt
    # the failure it exists to prevent: an answer that reports what the passages
    # say without ever committing, which the extractor reads as a refusal
    assert "reads as a refusal" in prompt
    assert "FIRST sentence must be that value" in prompt


def test_system_prompt_asks_for_a_bare_option_label():
    # "C) Online" fails the benchmark's exact match where "C" passes, so the
    # label must be given alone
    prompt = " ".join(gen._SYSTEM_PROMPT.split())
    assert "label ALONE" in prompt and '"C", not "C) Online"' in prompt


def test_the_cannot_be_determined_exception_follows_the_anti_hedging_rule():
    """Q75 was tagged unanswerable, yet the system opened with "Post-hoc correction"
    and only later admitted the sources could not settle it -- the extractor read the
    commitment. The exception must sit immediately after the rule that caused it, so
    the model sees both together."""
    prompt = gen._SYSTEM_PROMPT
    lead = prompt.index("LEAD WITH THE DIRECT ANSWER")
    exception = prompt.index("EXCEPTION, and it overrides")
    options = prompt.index("If the question offers labelled options")
    assert lead < exception < options
    flat = " ".join(prompt.split())
    assert 'FIRST sentence must be exactly "Cannot be determined from the provided sources."' in flat
    assert "Never lead with a value you go on to disown" in flat


def test_the_harness_reads_the_required_opening_as_a_decline():
    from verirag.eval.harness import classify_outcome

    answer = ("Cannot be determined from the provided sources. Table 2 lists post-hoc "
              "work but the passages give no counts for the other strategies [1].")
    # without this the answer scores 1.0 yet is classified "answered", inflating the
    # false-positive rate on exactly the questions the exception is meant to fix
    assert classify_outcome(False, answer, "verified") == "refused"
    assert classify_outcome(False, answer, "partial") == "partial"  # pipeline signal still wins
