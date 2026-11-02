"""Reusable evaluation harness.

Takes a question-set JSON and an eval store, runs every question through
``run_query``, and produces per-question traces, computed metrics, a markdown
report, and -- given a baseline run -- an automatic before/after diff.

Question-set format (``eval_data/*.json``)::

    {
      "name": "queensland_stratified",
      "corpus": "/path/to/papers",
      "questions": [
        {
          "id": 1,
          "group": "single_hop_factoid",     # the stratum
          "question": "How many parameters does GFM-RAG have?",
          "ground_truth": "8M parameters.",
          "expected_verdict": "answer",      # "answer" | "decline"
          "must_contain": [["8M", "8 million"]]   # optional, see below
        }
      ]
    }

``must_contain`` is how correctness is scored WITHOUT a judge model: each inner
list is a set of acceptable surface forms and every outer entry must match one
of them. It is a deliberately lexical proxy -- a correct answer phrased in an
unanticipated way counts as incorrect -- so a flagged hallucination should
always be read before being believed. Questions without ``must_contain`` are
counted as answered but not scored for correctness.

Outcomes are FOUR, not two: ``answered`` (a fully verified answer), ``partial``
(Gate 2 verified some claims and not others; the answer is returned with an
explicit Verified facts list), ``refused`` (a prose decline) and ``abstained`` (a
gate declined). ``partial`` is taken from the pipeline's own ``outcome`` rather
than re-derived from the text, so the harness cannot disagree with the system.

STRICT is the headline wherever a number could be flattered by an unverified
statement. For a ``partial`` answer, correctness (``must_contain``) is checked
against the Verified facts ONLY; the whole-response check is kept as
``correct_lenient``, an upper bound. The same two-pass idea scores the
MMLongBench-Doc benchmark (see eval/mmlongbench_scorer.py).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from verirag.generation.partial import split_verified_facts

logger = logging.getLogger(__name__)

# A decline can arrive two ways: a gate abstains, or generation explicitly
# says the sources do not cover the question. Both are refusals, not answers.
_REFUSAL_MARKERS = (
    "do not address",
    "does not address",
    "do not contain",
    "does not contain",
    "do not describe",
    "not addressed",
    "no information",
    "do not provide",
    "do not give",
    "do not state",
    # the generator's required opening when it concludes the sources cannot settle
    # the question (see generation/generator.py); without it such an answer is
    # classified "answered" and counted as a false positive despite scoring 1.0
    "cannot be determined",
    "do not fully address",
)


@dataclass
class EvalMetrics:
    """Programmatically computed, never hand-counted.

    How ``partial`` is counted, and why (it is a real third outcome, so every
    rate says where it sits):

    * ``recall`` counts only FULLY verified answers. A partial answer carries
      statements the system did not vouch for, so crediting it would flatter the
      headline; ``recall_incl_partial`` is the upper bound.
    * ``false_positive_rate`` counts ANY answer returned where a decline was
      expected, partial included: handing the reader content they should not have
      been given is a cost even when it is labelled. ``false_positive_rate_
      verified_only`` is the lower bound, and ``partial_on_decline`` is the
      accepted partial-mode tradeoff, tracked on its own so it is never hidden
      inside either number.
    * ``refusal_precision`` and ``hallucination_rate`` are unchanged: a partial
      answer is not a decline, and the hallucination invariant is about fully
      verified answers. A partial answer whose VERIFIED FACTS are wrong is listed
      in ``partial_wrong`` instead.

    Both choices are the pessimistic reading (strict, like the benchmark
    headline), with the optimistic reading reported beside it.
    """

    total: int
    answered: int
    declined: int
    # Of questions that SHOULD be answered, how many were answered (verified in full).
    recall: float
    # Of the times the system declined, how often declining was right.
    refusal_precision: float
    # Of questions that should have been declined, how many got an answer
    # (full or partial). HEADLINE; see the class docstring.
    false_positive_rate: float
    # Of scored fully-verified answers, how many were factually wrong. MUST be 0.
    hallucination_rate: float
    scored_answers: int
    hallucinations: list[int] = field(default_factory=list)
    avg_regenerations: float = 0.0
    by_group: dict[str, dict[str, Any]] = field(default_factory=dict)
    # --- the third outcome ---
    partial: int = 0
    partial_rate: float = 0.0  # partial / total
    partial_on_answerable: int = 0
    partial_on_decline: int = 0
    recall_incl_partial: float = 0.0
    false_positive_rate_verified_only: float = 0.0
    partial_wrong: list[int] = field(default_factory=list)


def load_question_set(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    for question in data["questions"]:
        for required in ("id", "group", "question", "expected_verdict"):
            if required not in question:
                raise ValueError(f"question {question.get('id')} missing {required!r}")
        if question["expected_verdict"] not in ("answer", "decline"):
            raise ValueError(
                f"question {question['id']}: expected_verdict must be "
                f"'answer' or 'decline'"
            )
    return data


def classify_outcome(
    abstained: bool, answer: str | None, pipeline_outcome: str | None = None
) -> str:
    """answered | partial | refused | abstained.

    ``partial`` comes straight from ``pipeline_outcome`` (the pipeline's own
    ``VeriRAGResult.outcome``): the harness reuses the signal that triggered
    partial mode instead of guessing from the text, so a partial answer whose
    opening happens to be a refusal ("The passages do not give ...") or a premise
    correction ("... appears in the QpiGNN paper, not in GFM-RAG") is still
    ``partial``, not forced into answered/declined by phrase detection.

    Otherwise: a prose refusal LEADS with its declination ("The provided documents
    do not address this question."), because the generator is instructed to say so
    directly and stop. Scanning the WHOLE answer for a refusal marker instead
    misclassifies a substantive, verified answer that merely uses a negation in
    passing ("GFM-RAG does not state a GPU-hour cost, but reports ...") as a
    refusal -- which wrongly drops it out of recall AND out of correctness
    scoring (correctness is only computed for `answered`, so a mislabelled
    answer can also hide a hallucination). So the marker check is restricted to
    the first sentence: only a leading declination counts as a refusal.
    """
    if pipeline_outcome == "partial" and not abstained:
        return "partial"
    if abstained:
        return "abstained"
    text = (answer or "").strip()
    if not text:
        return "answered"
    first_sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0].lower()
    if any(marker in first_sentence for marker in _REFUSAL_MARKERS):
        return "refused"
    return "answered"


def check_must_contain(
    answer: str | None, must_contain: list[list[str]] | None
) -> bool | None:
    """True/False if scoreable, None if the question defines no expectation."""
    if not must_contain:
        return None
    haystack = re.sub(r"\s+", " ", (answer or "")).lower()
    for alternatives in must_contain:
        forms = alternatives if isinstance(alternatives, list) else [alternatives]
        if not any(str(form).lower() in haystack for form in forms):
            return False
    return True


def compute_metrics(records: list[dict]) -> EvalMetrics:
    total = len(records)
    answered = [r for r in records if r["outcome"] == "answered"]
    partial = [r for r in records if r["outcome"] == "partial"]
    declined = [r for r in records if r["outcome"] in ("refused", "abstained")]

    want_answer = [r for r in records if r["expected_verdict"] == "answer"]
    want_decline = [r for r in records if r["expected_verdict"] == "decline"]

    def share(part: list[dict], whole: list[dict]) -> float:
        return len(part) / len(whole) if whole else 0.0

    recall = share([r for r in want_answer if r["outcome"] == "answered"], want_answer)
    recall_incl_partial = share(
        [r for r in want_answer if r["outcome"] in ("answered", "partial")], want_answer
    )
    refusal_precision = share(
        [r for r in declined if r["expected_verdict"] == "decline"], declined
    )
    false_positive_rate = share(
        [r for r in want_decline if r["outcome"] in ("answered", "partial")], want_decline
    )
    false_positive_rate_verified_only = share(
        [r for r in want_decline if r["outcome"] == "answered"], want_decline
    )

    scored = [r for r in answered if r["correct"] is not None]
    hallucinations = [r["id"] for r in scored if r["correct"] is False]
    hallucination_rate = len(hallucinations) / len(scored) if scored else 0.0
    partial_wrong = [r["id"] for r in partial if r.get("correct") is False]

    by_group: dict[str, dict[str, Any]] = {}
    for record in records:
        bucket = by_group.setdefault(
            record["group"],
            {"total": 0, "answered": 0, "partial": 0, "declined": 0, "wrong": 0},
        )
        bucket["total"] += 1
        if record["outcome"] == "answered":
            bucket["answered"] += 1
            if record["correct"] is False:
                bucket["wrong"] += 1
        elif record["outcome"] == "partial":
            bucket["partial"] += 1
        else:
            bucket["declined"] += 1

    return EvalMetrics(
        total=total,
        answered=len(answered),
        declined=len(declined),
        recall=recall,
        refusal_precision=refusal_precision,
        false_positive_rate=false_positive_rate,
        hallucination_rate=hallucination_rate,
        scored_answers=len(scored),
        hallucinations=hallucinations,
        avg_regenerations=(
            sum(r["regeneration_count"] for r in records) / total if total else 0.0
        ),
        by_group=by_group,
        partial=len(partial),
        partial_rate=len(partial) / total if total else 0.0,
        partial_on_answerable=len([r for r in want_answer if r["outcome"] == "partial"]),
        partial_on_decline=len([r for r in want_decline if r["outcome"] == "partial"]),
        recall_incl_partial=recall_incl_partial,
        false_positive_rate_verified_only=false_positive_rate_verified_only,
        partial_wrong=partial_wrong,
    )


def configure_store(store_path: str | Path) -> Path:
    """Point every store at an isolated eval location. Call before running."""
    from verirag.config import settings

    store = Path(store_path)
    store.mkdir(parents=True, exist_ok=True)
    settings.chroma_persist_dir = str(store / "chroma")
    settings.bm25_corpus_path = str(store / "bm25_corpus.jsonl")
    settings.metrics_db_path = str(store / "metrics.db")
    return store


def run_question_set(
    question_set: dict,
    store_path: str | Path,
    only_ids: list[int] | None = None,
    corpus_only: bool = True,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Run every (selected) question and return the full result payload.

    With ``checkpoint_path``, the payload so far is rewritten after EVERY
    question. A long run that is interrupted -- a timeout, a laptop shutdown --
    then still leaves usable results instead of nothing, which has already
    cost this project two complete runs.
    """
    import dataclasses as _dc

    from verirag.generation.generator import AnswerFormat
    from verirag.graph.build import run_query
    from verirag.retrieval.vector_store import list_ingested_documents

    store = configure_store(store_path)
    questions = question_set["questions"]
    if only_ids:
        wanted = set(only_ids)
        questions = [q for q in questions if q["id"] in wanted]

    documents = list_ingested_documents()
    print(f"store: {store}  documents: {len(documents)}  "
          f"questions: {len(questions)}", flush=True)

    records: list[dict] = []
    for question in questions:
        started = time.perf_counter()
        result = run_query(
            question["question"], AnswerFormat.CONCISE, corpus_only=corpus_only
        )
        outcome = classify_outcome(result.abstained, result.answer, result.outcome)
        must = question.get("must_contain")
        if outcome == "partial":
            # STRICT: only the Verified facts may satisfy must_contain; the
            # whole-response check is the lenient upper bound.
            _, facts = split_verified_facts(result.answer or "")
            correct = check_must_contain(facts, must)
            correct_lenient = check_must_contain(result.answer, must)
        else:
            correct = check_must_contain(result.answer, must) if outcome == "answered" else None
            correct_lenient = correct
        meets = (
            outcome == "answered"
            if question["expected_verdict"] == "answer"
            else outcome in ("refused", "abstained")
        )

        records.append(dict(
            id=question["id"],
            group=question["group"],
            question=question["question"],
            ground_truth=question.get("ground_truth"),
            expected_verdict=question["expected_verdict"],
            must_contain=question.get("must_contain"),
            outcome=outcome,
            meets_expectation=meets,
            correct=correct,
            correct_lenient=correct_lenient,
            pipeline_outcome=result.outcome,
            verified_facts=[f.text for f in result.verified_facts],
            answer=result.answer,
            abstention_reason=result.abstention_reason,
            regeneration_count=result.regeneration_count,
            citations=[_dc.asdict(c) | {"parent_text": None}
                       for c in result.citations],
            trace=result.trace,
            elapsed_s=round(time.perf_counter() - started, 1),
        ))

        if checkpoint_path is not None:
            partial = dict(
                name=question_set.get("name", "eval"),
                date=date.today().isoformat(),
                store=str(store),
                corpus=question_set.get("corpus"),
                documents=documents,
                questions=records,
                metrics=dataclasses.asdict(compute_metrics(records)),
                complete=False,
                progress=f"{len(records)}/{len(questions)}",
            )
            Path(checkpoint_path).write_text(
                json.dumps(partial, indent=2, default=str)
            )

        flag = "ok" if meets else "MISMATCH"
        extra = "" if correct is None else (" correct" if correct else " WRONG")
        if outcome == "partial":
            extra += " (strict)"
        print(f"  [Q{question['id']:>3}] {question['group']:22} "
              f"{outcome:10} want={question['expected_verdict']:7} "
              f"{flag}{extra}  ({records[-1]['elapsed_s']}s)", flush=True)

    metrics = compute_metrics(records)
    return dict(
        complete=True,
        name=question_set.get("name", "eval"),
        date=date.today().isoformat(),
        store=str(store),
        corpus=question_set.get("corpus"),
        documents=documents,
        questions=records,
        metrics=dataclasses.asdict(metrics),
    )


def diff_against_baseline(
    records: list[dict], baseline: dict
) -> list[dict]:
    """Per-question before/after rows, matched on question id."""
    before_by_id = {r["id"]: r for r in baseline.get("questions", [])}
    rows: list[dict] = []
    for record in records:
        before = before_by_id.get(record["id"])
        if before is None:
            rows.append(dict(id=record["id"], group=record["group"],
                             before="(new)", after=record["outcome"],
                             change="added"))
            continue
        # Tolerate earlier report schemas, which stored the outcome as "after".
        before_outcome = (
            before.get("outcome")
            or before.get("after")
            or before.get("verdict")
            or "?"
        )
        if before_outcome == record["outcome"]:
            change = "unchanged"
        elif record["outcome"] == "answered":
            change = "now answers"
        elif record["outcome"] == "partial":
            change = "now partial"
        elif before_outcome == "answered":
            change = "now declines"
        else:
            change = "changed"
        rows.append(dict(id=record["id"], group=record["group"],
                         before=before_outcome, after=record["outcome"],
                         change=change))
    return rows


def render_markdown(payload: dict, baseline: dict | None = None) -> str:
    """Markdown report. Flags a non-zero hallucination rate loudly."""
    metrics = payload["metrics"]
    records = payload["questions"]
    lines: list[str] = []

    lines.append(f"# Eval: {payload['name']} — {payload['date']}")
    lines.append("")
    lines.append(f"Store: `{payload['store']}` · "
                 f"{len(payload.get('documents') or [])} documents · "
                 f"{metrics['total']} questions · `corpus_only=True`")
    lines.append("")

    if metrics["hallucination_rate"] > 0:
        lines.append("## 🚨 HALLUCINATION RATE IS NOT ZERO")
        lines.append("")
        lines.append(
            f"**{metrics['hallucination_rate']:.1%} of scored answers were "
            f"factually wrong** (questions "
            f"{metrics['hallucinations']}). The system's core invariant is "
            f"that it verifies or abstains — investigate before trusting any "
            f"other number in this report."
        )
        lines.append("")
    else:
        lines.append("**Hallucination rate: 0.0%** — no scored answer was "
                     "factually wrong.")
        lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| questions | {metrics['total']} |")
    lines.append(f"| answered (fully verified) | {metrics['answered']} |")
    lines.append(f"| partial (some claims unverified) | {metrics.get('partial', 0)} |")
    lines.append(f"| declined | {metrics['declined']} |")
    lines.append(f"| **recall** (answerable answered in full; partial not counted) | "
                 f"**{metrics['recall']:.1%}** |")
    lines.append(f"| recall incl. partial (upper bound) | "
                 f"{metrics.get('recall_incl_partial', metrics['recall']):.1%} |")
    lines.append(f"| **refusal precision** (declines that were right) | "
                 f"**{metrics['refusal_precision']:.1%}** |")
    lines.append(f"| **false positive rate** (should decline, got ANY answer incl. partial) | "
                 f"**{metrics['false_positive_rate']:.1%}** |")
    lines.append(f"| false positive rate, verified answers only (lower bound) | "
                 f"{metrics.get('false_positive_rate_verified_only', metrics['false_positive_rate']):.1%} |")
    lines.append(f"| partial on should-decline (accepted partial-mode tradeoff) | "
                 f"{metrics.get('partial_on_decline', 0)} |")
    lines.append(f"| partial on should-answer | {metrics.get('partial_on_answerable', 0)} |")
    lines.append(f"| **hallucination rate** (scored answers wrong) | "
                 f"**{metrics['hallucination_rate']:.1%}** "
                 f"({metrics['scored_answers']} scored) |")
    lines.append(f"| avg regenerations | {metrics['avg_regenerations']:.2f} |")
    lines.append("")

    lines.append("## By stratum")
    lines.append("")
    lines.append("| stratum | n | answered | partial | declined | wrong |")
    lines.append("|---|---|---|---|---|---|")
    for group, bucket in sorted(metrics["by_group"].items()):
        lines.append(f"| {group} | {bucket['total']} | {bucket['answered']} "
                     f"| {bucket.get('partial', 0)} | {bucket['declined']} "
                     f"| {bucket['wrong']} |")
    lines.append("")

    if baseline is not None:
        lines.append("## Before / after vs baseline")
        lines.append("")
        lines.append(f"Baseline: `{baseline.get('name', 'baseline')}` "
                     f"({baseline.get('date', 'unknown date')})")
        lines.append("")
        lines.append("| Q | stratum | before | after | change |")
        lines.append("|---|---|---|---|---|")
        for row in diff_against_baseline(records, baseline):
            lines.append(f"| {row['id']} | {row['group']} | {row['before']} "
                         f"| {row['after']} | {row['change']} |")
        lines.append("")

    lines.append("## Per question")
    lines.append("")
    lines.append("| Q | stratum | want | outcome | correct | regens | meets |")
    lines.append("|---|---|---|---|---|---|---|")
    for record in records:
        correct = ("—" if record["correct"] is None
                   else ("yes" if record["correct"] else "**NO**"))
        lines.append(
            f"| {record['id']} | {record['group']} | "
            f"{record['expected_verdict']} | {record['outcome']} | {correct} "
            f"| {record['regeneration_count']} | "
            f"{'✅' if record['meets_expectation'] else '❌'} |"
        )
    lines.append("")

    mismatches = [r for r in records if not r["meets_expectation"]]
    if mismatches:
        lines.append("## Mismatches")
        lines.append("")
        for record in mismatches:
            lines.append(f"**Q{record['id']}** ({record['group']}, wanted "
                         f"{record['expected_verdict']}, got "
                         f"{record['outcome']})")
            lines.append("")
            lines.append(f"- question: {record['question']}")
            lines.append(f"- ground truth: {record['ground_truth']}")
            detail = record["abstention_reason"] or record["answer"] or ""
            lines.append(f"- system: {detail[:300]}")
            lines.append("")

    return "\n".join(lines)


def write_outputs(
    payload: dict, out_dir: str | Path, baseline: dict | None = None
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{payload['name']}_{payload['date']}"
    traces = out / f"{stem}.json"
    report = out / f"{stem}.md"
    traces.write_text(json.dumps(payload, indent=2, default=str))
    report.write_text(render_markdown(payload, baseline))
    return traces, report
