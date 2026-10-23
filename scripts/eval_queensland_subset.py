"""Targeted QueensLand re-run: did the false negatives convert, and did we
break any correct refusal?

Reuses the EXACT question text and ground-truth labels from
scripts/eval_queensland_corpus.py, so before/after is directly comparable,
and runs only a prioritised subset against the same isolated eval store.

Subset rationale (the full set is 21; running "every false negative plus every
correct abstention plus 3 easy ones" would be 20 of them, i.e. not a subset):
  A. false negatives to convert - the six named ones plus Q8, which is the
     only question that exercised the formula/unscorable claim path (Bug 3)
  B. ALL correct abstentions - a regression here is the single most important
     result, since it would mean trading a false negative for a false positive
  C. three previously-correct answers, one per distinct paper, as a regression
     check on the easy cases

Deliberately skipped: Q5, Q7, Q10, Q14, Q15, Q17 (further false negatives of
the same Gate 2 kind as Q2/Q6/Q11) and Q12 (a second G-Reasoner easy case).

    python scripts/eval_queensland_subset.py
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import date
from pathlib import Path

# Importing the main eval module points every store at eval_store/queensland
# and gives us the identical QUESTIONS list.
from scripts.eval_queensland_corpus import (  # noqa: E402
    CORPUS_DIR,
    EVAL_STORE,
    QUESTIONS,
    RESULTS_DIR,
    _corpus_files,
)
from verirag.generation.generator import AnswerFormat  # noqa: E402
from verirag.graph.build import run_query  # noqa: E402
from verirag.ingest import ingest_file  # noqa: E402
from verirag.observability.metrics import summarize_metrics  # noqa: E402
from verirag.retrieval.vector_store import list_ingested_documents  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# What the ORIGINAL 2026-10-23 eval produced, from its report.
#   verdict: what the system did
#   correct: whether that was right
#   want:    required behaviour now ("answer" = substantive answer,
#            "decline" = abstain or an explicit "sources don't cover this")
BEFORE = {
    1:  dict(group="C", verdict="answered",  correct=True,  want="answer"),
    2:  dict(group="A", verdict="abstained", correct=False, want="answer"),
    3:  dict(group="C", verdict="answered",  correct=True,  want="answer"),
    4:  dict(group="A", verdict="abstained", correct=False, want="answer"),
    6:  dict(group="A", verdict="abstained", correct=False, want="answer"),
    8:  dict(group="A", verdict="abstained", correct=False, want="answer"),
    9:  dict(group="C", verdict="answered",  correct=True,  want="answer"),
    11: dict(group="A", verdict="abstained", correct=False, want="answer"),
    13: dict(group="A", verdict="partial",   correct=False, want="answer"),
    16: dict(group="B", verdict="abstained", correct=True,  want="decline"),
    18: dict(group="A", verdict="partial",   correct=False, want="answer"),
    19: dict(group="B", verdict="abstained", correct=True,  want="decline"),
    20: dict(group="B", verdict="abstained", correct=True,  want="decline"),
    21: dict(group="B", verdict="refused",   correct=True,  want="decline"),
}

_REFUSAL_MARKERS = (
    "do not address",
    "does not address",
    "do not contain",
    "does not contain",
    "do not describe",
    "not addressed",
    "no information",
    "do not provide",
    "do not state",
)


def _classify(result) -> str:
    """answered / refused / abstained -- a refusal is a decline, not an answer."""
    if result.abstained:
        return "abstained"
    text = (result.answer or "").lower()
    if any(marker in text for marker in _REFUSAL_MARKERS):
        return "refused"
    return "answered"


def _rule(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    by_id = {item["id"]: item for item in QUESTIONS}
    subset = sorted(BEFORE)

    _rule("STEP 1 - complete the interrupted ingestion")
    print(f"corpus: {CORPUS_DIR}")
    for path in _corpus_files():
        result = ingest_file(path)
        state = "skipped (already ingested)" if result.skipped else (
            f"text={result.num_text_chunks} fig={result.num_figure_chunks}"
        )
        print(f"  {path.name}: {state}")
    docs = list_ingested_documents()
    print(f"\ndistinct documents now in eval store: {len(docs)}")
    for doc in docs:
        print(f"  {Path(doc['source_path']).name}  group={doc['duplicate_group_id']}")

    _rule(f"STEP 2 - run the {len(subset)}-question subset (corpus_only=True)")
    records = []
    for qid in subset:
        item = by_id[qid]
        before = BEFORE[qid]
        print("-" * 78)
        print(f"[Q{qid}] group {before['group']} | before: {before['verdict']}"
              f" ({'correct' if before['correct'] else 'WRONG'})"
              f" | want: {before['want']}")
        print(f"  {item['q']}")

        result = run_query(item["q"], AnswerFormat.CONCISE, corpus_only=True)
        after = _classify(result)
        declined = after in ("abstained", "refused")
        ok = declined if before["want"] == "decline" else (after == "answered")

        print(f"  -> {after} (regens={result.regeneration_count}) "
              f"{'OK' if ok else 'MISMATCH'}")
        if result.abstained:
            print(f"     reason: {(result.abstention_reason or '')[:200]}")
        else:
            print(f"     answer: {(result.answer or '')[:280]}")
            for c in result.citations:
                label = Path(str(c.source_path)).name
                print(f"       [{c.number}] {label} "
                      f"({len(c.contributing_chunk_ids)} chunks)")

        records.append(dict(
            id=qid, group=before["group"], question=item["q"],
            ground_truth=item["truth"], before=before, after=after,
            matches_requirement=ok,
            answer=result.answer, abstention_reason=result.abstention_reason,
            regeneration_count=result.regeneration_count,
            citations=[dataclasses.asdict(c) | {"parent_text": None}
                       for c in result.citations],
            trace=result.trace,
        ))

    _rule("BEFORE / AFTER")
    print(f"{'Q':>3}  {'grp':3}  {'before':10} {'after':10} {'want':8} result")
    for rec in records:
        print(f"{rec['id']:>3}  {rec['group']:3}  {rec['before']['verdict']:10} "
              f"{rec['after']:10} {rec['before']['want']:8} "
              f"{'OK' if rec['matches_requirement'] else 'MISMATCH'}")

    converted = [r for r in records if r["group"] == "A" and r["after"] == "answered"]
    still_failing = [r for r in records
                     if r["group"] == "A" and r["after"] != "answered"]
    regressions = [r for r in records
                   if r["group"] == "B" and r["after"] == "answered"]
    easy_broken = [r for r in records
                   if r["group"] == "C" and r["after"] != "answered"]

    _rule("VERDICT")
    group_a = [r for r in records if r["group"] == "A"]
    group_b = [r for r in records if r["group"] == "B"]
    group_c = [r for r in records if r["group"] == "C"]
    print(f"A. false negatives converted : {len(converted)}/{len(group_a)}"
          f"  (still failing: {[r['id'] for r in still_failing]})")
    print(f"B. correct refusals held     : "
          f"{len(group_b) - len(regressions)}/{len(group_b)}")
    print(f"C. easy cases still correct  : "
          f"{len(group_c) - len(easy_broken)}/{len(group_c)}"
          f"  (broken: {[r['id'] for r in easy_broken]})")
    print(f"avg regenerations            : "
          f"{sum(r['regeneration_count'] for r in records) / len(records):.2f}")

    if regressions:
        print("\n*** PRECISION REGRESSION ***")
        print("These previously-correct refusals now produce an answer:")
        for rec in regressions:
            print(f"  Q{rec['id']}: {rec['question']}")
            print(f"    {(rec['answer'] or '')[:300]}")
    else:
        print("\nNo precision regression: every previously-correct refusal "
              "still declines.")

    stamp = date.today().isoformat()
    out = RESULTS_DIR / f"queensland_subset_eval_{stamp}.json"
    out.write_text(json.dumps(dict(
        date=stamp, eval_store=str(EVAL_STORE), subset=subset,
        documents=docs, questions=records,
        metrics=dataclasses.asdict(summarize_metrics()),
    ), indent=2, default=str))
    print(f"\ntraces written to {out}")


if __name__ == "__main__":
    main()
