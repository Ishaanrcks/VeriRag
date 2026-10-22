"""Regression test for claim extraction on long inputs.

Motivation: a whole-document summary used to blow the fixed 2000-token
extraction budget, the truncated JSON failed to parse, and the fallback
collapsed the entire answer into ONE mega-claim. A single claim spanning
thousands of characters can never be entailed by one ~500-token chunk, so
Gate 2 rejected genuinely well-grounded summaries. Confirmed live with
"JSON decode error: Expecting ',' delimiter: line 200 column 25 (char 7623)".

Part 1 checks the truncation-recovery logic deterministically (no API).
Part 2 runs a real long multi-paragraph summary through extract_claims().

    python scripts/test_claim_extraction.py
"""

from __future__ import annotations

import logging

from verirag.critic.claims import (
    _extraction_max_tokens,
    _recover_complete_objects,
    extract_claims,
)
from verirag.generation.generator import (
    AnswerFormat,
    Citation,
    GeneratedAnswer,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# A long, multi-paragraph summary in the shape the summarization path emits.
LONG_SUMMARY = """\
**What the document is**

- It is the paper "Walking a Tightrope - Evaluating Large Language Models in
  High-Risk Domains" by Chia-Chien Hung and colleagues at NEC Laboratories
  Europe [1].
- It studies instruction-tuned LLMs in high-risk domains, focusing on factual
  accuracy and safety adherence [1].
- Experiments cover six NLP datasets spanning question answering and
  summarization in two high-risk domains: legal and medical [1].

**Motivation and framing**

- LLMs tend to generate plausible but erroneous information, which is
  particularly detrimental in high-risk domains [1].
- The EU AI Act is expected to be the first worldwide law regulating AI
  deployment in the EU, imposing strict obligations on high-risk systems [1].
- LLMs are expected to fall into the high-risk category in specific domains
  such as medical and legal [1].

**Data and setup**

- Instruction-tuning datasets were built for legal and medical domains with
  13K instructions per domain for fair comparison [1].
- Medical data was downsampled to match the scarcer legal instance counts [1].
- Legal instruction data came from BillSum, CaseHold, LegalAdviceReddit and
  LawStackExchange; medical from PubMedQA, RCTSum, MedQA and HealthCareMagic [1].
- Factuality was measured with QAFactEval and UniEval; safety with Detoxify
  and SafetyKit [1].
- Models evaluated were GPT4ALL-J (6B), GPT4ALL-MPT (7B) and GPT-3.5-turbo [1].
- Training used a single Nvidia Tesla V100 with 32GB VRAM, 5 epochs, batch
  size 64, learning rate 1e-5 and max sequence length 1024 [1].

**Evaluation datasets and tasks**

- Legal evaluation used BillSum for summarization (100 instances), CaseHold
  for multiple-choice QA (1,000 instances) and LawStackExchange for free-form
  QA (989 instances) [1].
- Medical evaluation used PubMedQA for reasoning QA (250 instances), RCTSum
  for summarization (100 instances) and iCliniq for free-form QA (1,000
  instances) [1].
- All selected evaluation datasets were chosen to be commercially
  applicable [1].
- The evaluation spans multiple-choice QA, free-form QA, reasoning QA and
  long-document summarization [1].
- The overall evaluation approach aligns with the AuditNLG library [1].

**Stated contributions**

- A robust evaluation of out-of-the-box and instruction-tuned LLM outputs
  across six datasets in two high-risk domains for both safety and
  factuality [1].
- A qualitative investigation of the shortcomings of existing metrics [1].
- A discussion of open challenges for trusting LLM generalization in
  high-risk domains [1].
- Advocacy for human-centric NLP systems that give final control to human
  users for trustworthy high-risk applications [1].

**Findings**

- Instruction-tuning on domain data improved factuality scores relative to
  the out-of-the-box baselines [1].
- Existing automatic metrics were found inadequate for judging high-risk
  outputs, motivating a qualitative analysis [1].
- The authors advocate human-centric systems that leave final control with
  the human user [1].
- The study builds on Hupkes et al. (2022), examining generalization to new
  high-risk domains and to new tasks with and without instruction-tuning [1].
- Robustness is acknowledged as a relevant factor but was not prioritized in
  this evaluation [1].
- Instruction-style templates were manually composed for each task, each
  containing an instruction describing the task to be performed [1].

**Limits of the provided context**

- The passages do not describe deployment results outside the two studied
  domains [1].
- The passages do not report per-dataset numeric scores for every metric [1].
"""

# Deliberately truncated JSON: the array never closes, the final object is
# cut mid-value, and the last "]" belongs to a nested cited_numbers array --
# exactly the shape that used to defeat parsing.
TRUNCATED_JSON = """[
  {"text": "First complete claim.", "claim_type": "factual", "cited_numbers": [1]},
  {"text": "Second complete claim with a } brace in the text.", "claim_type": "factual", "cited_numbers": [1, 2]},
  {"text": "Third complete claim, with an escaped \\" quote.", "claim_type": "absence", "cited_numbers": []},
  {"text": "Fourth claim that got cut off mid-"""


def _rule(title: str) -> None:
    print("=" * 76)
    print(title)
    print("=" * 76)


def main() -> None:
    _rule("PART 1 - truncation recovery (deterministic, no API)")
    recovered = _recover_complete_objects(TRUNCATED_JSON)
    print(f"  complete objects recovered: {len(recovered)} (expected 3)")
    for obj in recovered:
        print(f"    {obj[:72]}...")
    print(f"  RESULT recovers complete objects only : {len(recovered) == 3}")
    print(f"  RESULT ignores the cut-off final object: "
          f"{'got cut off' not in ' '.join(recovered)}")

    print("  budget scaling:")
    for length in (300, 700, 6000, 20000, 60000):
        print(f"    answer {length:6} chars -> max_tokens="
              f"{_extraction_max_tokens('x' * length)}")
    print(f"  RESULT long summary gets a bigger budget than the old 2000: "
          f"{_extraction_max_tokens(LONG_SUMMARY) > 2000}")

    _rule("PART 2 - real extraction on a long multi-paragraph summary")
    print(f"  summary length: {len(LONG_SUMMARY)} chars")
    print(f"  extraction budget: {_extraction_max_tokens(LONG_SUMMARY)} tokens")

    answer = GeneratedAnswer(
        answer_text=LONG_SUMMARY,
        citations=[
            Citation(
                number=1,
                parent_text="(source document)",
                source_path="2311.14966v1.pdf",
                doc_id="2311.14966v1-bc1931d80170",
            )
        ],
        format_used=AnswerFormat.DETAILED,
        hits_used=[],
    )

    claims = extract_claims(answer)
    print(f"  claims extracted: {len(claims)}")
    by_type: dict[str, int] = {}
    for claim in claims:
        by_type[claim.claim_type] = by_type.get(claim.claim_type, 0) + 1
    print(f"  by claim_type: {by_type}")
    for i, claim in enumerate(claims[:6], start=1):
        print(f"    [{i}] ({claim.claim_type}, cites {claim.cited_numbers}) "
              f"{claim.text[:88]}")
    if len(claims) > 6:
        print(f"    ... and {len(claims) - 6} more")

    collapsed = len(claims) == 1 and len(claims[0].text) > 500
    print(f"  RESULT did NOT collapse into one mega-claim: {not collapsed}")
    print(f"  RESULT produced many atomic claims (>5)    : {len(claims) > 5}")
    print(f"  RESULT every claim is short enough to check: "
          f"{all(len(c.text) < 500 for c in claims)}")

    _rule("Done.")


if __name__ == "__main__":
    main()
