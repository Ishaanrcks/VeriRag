"""Table-aware claim verification (critic/table_claims.py + its Gate 2 route).

Built on question #8: "West Germany at 29% for Both (VOL)" is a true cell of the
Pew table, yet NLI scored such claims 0.0057-0.49 against its markdown. Tests
stub NLI to ZERO wherever a table route must stand on its own, so a pass cannot
come from entailment.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import verirag.critic.claims as claims_mod
from verirag.critic import gate2_groundedness as g2
from verirag.critic.table_claims import parse_markdown_table, verify_numeric_claim
from verirag.generation.generator import Citation, GeneratedAnswer
from verirag.retrieval.types import SearchHit

Q8 = json.loads((Path(__file__).parent / "fixtures" / "q8_false_hedge.json").read_text())
# The real Pew table as first extracted (spacer columns), and as cleaned by the
# empty-column drop. Values sit one column off their headers in both.
TABLE = Q8["table"]["text"]
TABLE_CLEAN = (
    "|  | US . . |  | Both(VOL) | Russia |\n| --- | --- | --- | --- | --- |\n"
    "| West |  | 43% | 29% | 21% |\n|  |  |  |  |  |\n| East | 23 | 36 |  | 38 |"
)


# --------------------------------------------------------------------------
# the matcher on its own
# --------------------------------------------------------------------------
VERIFIED = [
    'A chart lists West Germany at 29% for "Both (VOL)".',            # #8's claim
    'Among West Germans, 29% volunteered "Both."',
    "In the chart, 21% of West Germans choose Russia.",
    "A chart lists West Germany at 43% for the U.S.",                  # value shifted right of its header
    "East Germans volunteered Both at 36%.",                           # value shifted left of its header
    "23% of East Germans say the U.S.",
    "West Germans: 43% U.S., 29% Both and 21% Russia.",                # every number paired
]


@pytest.mark.parametrize("table", [TABLE, TABLE_CLEAN], ids=["spacer-columns", "cleaned"])
@pytest.mark.parametrize("claim", VERIFIED)
def test_correctly_paired_claims_verify(claim, table):
    assert verify_numeric_claim(claim, table)


NOT_VERIFIED = [
    ("wrong row: 29 is West's, not East's", "East Germans at 29% for Both."),
    ("wrong column: 29 is under Both, not Russia", "West Germans at 29% for Russia."),
    ("wrong column: 21 is under Russia, not Both", "West Germans at 21% for Both."),
    ("43 is West/US; the shift repair must not move it to Both", "West Germans at 43% for Both."),
    ("close is not exact", "West Germans at 30% for Both."),
    ("number in no cell of that row", "West Germans at 38% for Russia."),
    ("no row named", "29% volunteered Both."),
    ("no column named", "29% of West Germans."),
    ("one wrong number spoils the claim", "West Germans: 29% Both and 55% Russia."),
    ("a comparative is not a lookup", "West Germans at 29% for Both is higher than East."),
    ("a negation is not a lookup", "West Germans did not say 29% for Both."),
    ("an aggregate is not a cell", "The total for West Germans across all answers is 93%."),
    ("no number at all", "West Germans mostly chose the U.S."),
]


@pytest.mark.parametrize("table", [TABLE, TABLE_CLEAN], ids=["spacer-columns", "cleaned"])
@pytest.mark.parametrize("why,claim", NOT_VERIFIED, ids=[w for w, _ in NOT_VERIFIED])
def test_wrongly_paired_claims_do_not_verify(why, claim, table):
    assert verify_numeric_claim(claim, table) is None


def test_a_header_with_two_equally_valid_sides_is_not_guessed():
    # 5 sits under a blank header with an empty, named column on BOTH sides
    table = "|  | Alpha |  | Beta |\n| --- | --- | --- | --- |\n| Row |  | 5 |  |"
    assert verify_numeric_claim("Row has 5 for Alpha.", table) is None
    assert verify_numeric_claim("Row has 5 for Beta.", table) is None


def test_parse_markdown_table():
    rows = parse_markdown_table(TABLE_CLEAN)
    assert rows[0] == ["", "US . .", "", "Both(VOL)", "Russia"]
    assert rows[1][:2] == ["West", ""] and len(rows) == 4  # separator row is dropped
    assert parse_markdown_table("Just a paragraph of text.") is None
    assert parse_markdown_table("| a | b |\n| c | d |") is None  # no separator row
    assert parse_markdown_table("| a \\| b | c |\n| --- | --- |\n| x | 1 |")[0][0] == "a | b"


# --------------------------------------------------------------------------
# Gate 2: the route, and the NLI path it must not weaken
# --------------------------------------------------------------------------
def _hit(chunk_id, text, content_type):
    return SearchHit(chunk_id=chunk_id, raw_text=text,
                     metadata={"content_type": content_type, "page": 15})


def _answer(hits):
    citation = Citation(number=1, parent_text="", source_path="p.pdf", doc_id="d", pages=[15],
                        contributing_chunk_ids=[h.chunk_id for h in hits],
                        content_types=sorted({h.metadata["content_type"] for h in hits}))
    return GeneratedAnswer(answer_text="a", citations=[citation], format_used=None, hits_used=hits)


@pytest.fixture
def nli_says(monkeypatch):
    """Stub NLI to a fixed entailment probability; returns the call log."""
    calls: list[str] = []

    def install(prob):
        def score(claim, premises):
            calls.append(claim)
            return [SimpleNamespace(entailment_prob=prob) for _ in premises]

        monkeypatch.setattr(g2, "score_claim_against_premises", score)
        monkeypatch.setattr(g2, "check_entailment_windowed",
                            lambda premise, claim: SimpleNamespace(entailment_prob=prob))
        return calls

    return install


def _verdicts(monkeypatch, hits, claim_texts):
    payload = json.dumps([dict(text=t, claim_type="factual", cited_numbers=[1]) for t in claim_texts])
    monkeypatch.setattr(claims_mod, "chat_completion", lambda **kw: payload)
    return g2.check_groundedness(_answer(hits))


def test_true_table_claim_is_verified_with_nli_saying_no(monkeypatch, nli_says):
    calls = nli_says(0.0)
    verdict = _verdicts(monkeypatch, [_hit("t", TABLE, "table")],
                        ['A chart lists West Germany at 29% for "Both (VOL)".'])
    claim = verdict.claims[0]
    assert (claim.verdict, claim.evidence_chunk_ids) == ("supported_presence", ["t"])
    assert verdict.overall_grounded and calls == []  # NLI was never needed


def test_q8s_three_real_claims_all_verify(monkeypatch, nli_says):
    # the claims question #8's answer made, which Gate 2 rejected for NLI < 0.5
    nli_says(0.0)
    verdict = _verdicts(monkeypatch, [_hit("t", TABLE, "table")], [
        "A chart lists West Germany at 43% for the U.S.",
        'A chart lists West Germany at 29% for "Both (VOL)".',
        "A chart lists West Germany at 21% for Russia.",
    ])
    assert [v.verdict for v in verdict.claims] == ["supported_presence"] * 3
    assert verdict.overall_grounded


def test_wrong_row_or_column_claim_is_not_verified(monkeypatch, nli_says):
    nli_says(0.0)
    verdict = _verdicts(monkeypatch, [_hit("t", TABLE, "table")], [
        'A chart lists East Germany at 29% for "Both (VOL)".',   # 29 belongs to West
        "A chart lists West Germany at 29% for Russia.",         # 29 belongs to Both
    ])
    assert [v.verdict for v in verdict.claims] == ["not_supported", "not_supported"]
    assert not verdict.overall_grounded


def test_a_claim_the_table_route_cannot_verify_still_gets_nli(monkeypatch, nli_says):
    calls = nli_says(0.95)  # NLI accepts it
    verdict = _verdicts(monkeypatch, [_hit("t", TABLE, "table")],
                        ["West Germans are twice as likely to prefer the U.S. as Russia."])
    assert verdict.claims[0].verdict == "supported_single"
    assert len(calls) == 1


def test_the_same_markdown_in_a_text_chunk_gets_no_table_route(monkeypatch, nli_says):
    nli_says(0.0)
    verdict = _verdicts(monkeypatch, [_hit("x", TABLE, "text")],
                        ['A chart lists West Germany at 29% for "Both (VOL)".'])
    assert verdict.claims[0].verdict == "not_supported"


def test_a_caption_chunk_cannot_use_the_table_route(monkeypatch, nli_says):
    nli_says(0.0)
    verdict = _verdicts(monkeypatch, [_hit("c", TABLE, "figure_caption")],
                        ['A chart lists West Germany at 29% for "Both (VOL)".'])
    assert verdict.claims[0].verdict == "not_supported"


def test_table_cited_alongside_text_still_verifies_from_the_table(monkeypatch, nli_says):
    nli_says(0.0)
    hits = [_hit("x", "Prose about West Germany and Russia.", "text"), _hit("t", TABLE, "table")]
    verdict = _verdicts(monkeypatch, hits, ['A chart lists West Germany at 29% for "Both (VOL)".'])
    assert verdict.claims[0].evidence_chunk_ids == ["t"]
