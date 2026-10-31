"""Full MMLongBench-Doc benchmark runner: question set -> pipeline -> scorer.

    pipeline (retrieval -> Gate 1 -> generate -> Gate 2, partial answers)
      -> LLM answer extraction -> vendored official scorer -> accuracy / F1

SEQUENTIAL BY DESIGN. One question at a time, in one process, so only one model
set is ever resident; parallel model-loading wedged this 8 GB machine. The run is
checkpointed after every question and resumable.

Two scores are reported for every run (see eval/mmlongbench_scorer.py):

* STRICT is the HEADLINE. For a partial answer the extractor reads only the
  Verified facts, so a value the system did not vouch for scores as "Not
  answerable".
* LENIENT reads the whole response. It is an UPPER BOUND that includes unverified
  statements, and it is noisier (the extractor is not deterministic).

The outcome metrics (recall, false positives, partial rates) come from
``harness.compute_metrics``, where ``partial`` is a first-class outcome.

PREFLIGHT. Every question names a PDF. A question whose document is not in the
store cannot be answered, so scoring it would count a missing file as a correct
decline on the unanswerable ones. The runner therefore refuses to start unless it
is told what to do about missing documents: ``only_ingested`` (run just the
questions whose document is present; how tiered ingestion is measured) or
``allow_missing`` (record them as skipped, excluded from every metric).

SCOPE. ``pooled`` (default) searches the whole store, which is what a real corpus
looks like and what corpus-size calibration needs. ``doc`` prefixes the question
with its filename, which the production document resolver turns into a retrieval
filter on that one document (an oracle-scoped upper bound).
"""

from __future__ import annotations

import collections
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

from verirag.eval.harness import classify_outcome, compute_metrics, configure_store
from verirag.eval.mmlongbench_scorer import (
    LENIENT_LABEL,
    STRICT_LABEL,
    extract_answer,
    load_extraction_prompt,
    make_extraction_client,
    pooled_metrics,
    score_response_dual,
)

DEFAULT_SUBSET = "eval_data/mmlongbench_subset200_v1canonical.json"
SCOPES = ("pooled", "doc")


class RunnerError(RuntimeError):
    """A refusal to start, with a message saying what to do about it."""


# --------------------------------------------------------------------------
# extraction client (shared with the pilot script)
# --------------------------------------------------------------------------
class CountingClient:
    """Wraps an OpenAI-compatible client, counting calls and recording errors."""

    def __init__(self, client) -> None:
        self._client = client
        self.calls = 0
        self.errors: list[str] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        try:
            return self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            self.errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")
            raise


def build_extractor(kind: str):
    """``(kind, model, counting_client)`` for ``deepseek`` or ``gemini``."""
    from verirag.config import settings

    if kind == "gemini":
        if not settings.gemini_api_key:
            raise RunnerError("GEMINI_API_KEY is not set")
        client = make_extraction_client(settings.gemini_api_key, settings.gemini_base_url)
        model = settings.gemini_extraction_model
    else:
        client = make_extraction_client(settings.deepseek_api_key, settings.deepseek_base_url)
        model = settings.deepseek_chat_model
    return kind, model, CountingClient(client)


def response_text(result) -> str:
    """What the extractor reads: the answer, or a plain statement of the abstention."""
    if not result.abstained and result.answer:
        return result.answer
    reason = result.abstention_reason or "the retrieved documents do not cover it"
    return f"I cannot answer this question from the provided documents. {reason}"


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------
@dataclass
class Plan:
    todo: list[dict]
    skipped: list[dict] = field(default_factory=list)  # questions whose document is missing
    docs_needed: list[str] = field(default_factory=list)
    docs_present: list[str] = field(default_factory=list)
    docs_missing: list[str] = field(default_factory=list)


def store_filenames(ingested: list[dict]) -> set[str]:
    """Filenames of the documents in the store (``source_path`` basenames)."""
    return {Path(str(d.get("source_path", ""))).name for d in ingested}


def make_plan(
    questions: list[dict],
    ingested: list[dict],
    *,
    only_ingested: bool = False,
    allow_missing: bool = False,
    ids: list[int] | None = None,
    limit: int | None = None,
) -> Plan:
    """Decide which questions to run; refuse if documents are missing and unhandled."""
    selected = [q for q in questions if not ids or q["id"] in set(ids)]
    present_names = store_filenames(ingested)
    needed = sorted({q["doc_id"] for q in selected})
    present = [d for d in needed if d in present_names]
    missing = [d for d in needed if d not in present_names]

    if missing and not (only_ingested or allow_missing):
        raise RunnerError(
            f"{len(missing)} of {len(needed)} documents needed by the selected "
            f"questions are not in the store (e.g. {missing[0]}). Ingest them, or "
            "pass only_ingested=True to run just the questions whose document is "
            "present, or allow_missing=True to record the rest as skipped."
        )
    todo = [q for q in selected if q["doc_id"] in present_names]
    skipped = [q for q in selected if q["doc_id"] not in present_names]
    if limit:
        todo = todo[:limit]
    return Plan(todo=todo, skipped=skipped, docs_needed=needed,
                docs_present=present, docs_missing=missing)


# --------------------------------------------------------------------------
# one question
# --------------------------------------------------------------------------
def query_text(row: dict, scope: str) -> str:
    """The question as sent to the pipeline; ``doc`` scope names the file so the
    production resolver restricts retrieval to it."""
    if scope == "doc":
        return f"{row['doc_id']} {row['question']}"
    return row["question"]


def _retrieved(result) -> list[dict]:
    """The chunks retrieval returned (doc, page, type, score), for retrieval metrics."""
    for event in getattr(result, "pipeline_log", None) or []:
        if event.stage == "retrieval":
            return [
                {k: c.get(k) for k in ("doc_name", "page", "type", "rerank_score")}
                for c in event.detail.get("chunks", [])
            ]
    return []


def process_question(
    row: dict,
    *,
    query_fn: Callable[[str], Any],
    extractor: Callable[[str, str], str],
    scope: str,
) -> dict:
    """Run one question through the pipeline and score it strict and lenient."""
    started = time.perf_counter()
    result = query_fn(query_text(row, scope))
    response = response_text(result)
    sample = dict(question=row["question"], answer=row["ground_truth"],
                  answer_format=row["answer_format"])
    dual = score_response_dual(sample, response, extractor)
    strict, lenient = dual["strict"], dual["lenient"]
    g1 = getattr(result, "gate1_verdict", None)
    g2 = getattr(result, "gate2_verdict", None)
    return dict(
        id=row["id"], doc_id=row["doc_id"], doc_type=row.get("doc_type"),
        group=row["group"], evidence_sources=row.get("evidence_sources", []),
        evidence_pages=row.get("evidence_pages", []), cross_page=row.get("cross_page"),
        answer_format=row["answer_format"], v2_status=row.get("v2_status", "unchanged"),
        expected_verdict=row["expected_verdict"], question=row["question"],
        ground_truth=row["ground_truth"],
        outcome=classify_outcome(result.abstained, result.answer, result.outcome),
        pipeline_outcome=result.outcome, abstained=result.abstained,
        abstention_reason=result.abstention_reason,
        answer=result.answer, response=response,
        verified_facts=[f.text for f in getattr(result, "verified_facts", [])],
        # STRICT is the headline; LENIENT is the upper bound
        pred=strict["pred"], score=strict["score"], extracted_res=strict["extracted_res"],
        lenient_pred=lenient["pred"], lenient_score=lenient["score"],
        partial=dual["partial"],
        correct=None, regeneration_count=result.regeneration_count,
        gate1=(f"{g1.verdict}/{g1.decision_path}/{g1.max_rerank_score:.3f}" if g1 else None),
        gate2_labels=dict(collections.Counter(c.verdict for c in g2.claims)) if g2 else {},
        retrieved=_retrieved(result),
        citations=[dict(doc_id=c.doc_id, pages=c.pages,
                        chunk_ids=c.contributing_chunk_ids)
                   for c in getattr(result, "citations", [])],
        scope=scope, elapsed_s=round(time.perf_counter() - started, 1),
    )


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def breakdowns(records: list[dict]) -> dict[str, dict[str, dict]]:
    """Strict and lenient accuracy per slice, the way the official report slices it
    (single-page / cross-page / unanswerable, evidence source, document type) plus
    this run's own: stratum, V2 status and outcome."""
    def answerability(r: dict) -> str:
        if r["expected_verdict"] == "decline":
            return "unanswerable"
        return "cross_page" if r.get("cross_page") else "single_page"

    dims: dict[str, Callable[[dict], list[str]]] = {
        "answerability": lambda r: [answerability(r)],
        "evidence_source": lambda r: list(r.get("evidence_sources") or ["(none)"]),
        "doc_type": lambda r: [r.get("doc_type") or "(unknown)"],
        "group": lambda r: [r["group"]],
        "v2_status": lambda r: [r.get("v2_status", "unchanged")],
        "outcome": lambda r: [r["outcome"]],
    }
    out: dict[str, dict[str, dict]] = {}
    for dim, keys in dims.items():
        buckets: dict[str, list[dict]] = collections.defaultdict(list)
        for r in records:
            for k in keys(r):
                buckets[k].append(r)
        out[dim] = {
            k: dict(n=len(rs), strict_accuracy=_mean([r["score"] for r in rs]),
                    lenient_accuracy=_mean([r["lenient_score"] for r in rs]))
            for k, rs in sorted(buckets.items())
        }
    return out


def summarise(records: list[dict]) -> dict:
    """Everything the report needs, from scored records (skipped ones excluded)."""
    strict = pooled_metrics([dict(r, pred=r["pred"], score=r["score"],
                                  answer=r["ground_truth"]) for r in records])
    lenient = pooled_metrics([dict(r, pred=r["lenient_pred"], score=r["lenient_score"],
                                   answer=r["ground_truth"]) for r in records])
    return dict(
        metrics_strict=strict,
        metrics_lenient=lenient,
        outcome_metrics=asdict(compute_metrics(records)),
        breakdowns=breakdowns(records),
    )


def render_report(payload: dict) -> str:
    """Markdown report: headline strict, lenient labelled as an upper bound."""
    s, l = payload["metrics_strict"], payload["metrics_lenient"]
    o, run = payload["outcome_metrics"], payload["run"]
    lines = [f"# MMLongBench-Doc — {payload['name']} — {payload['date']}", ""]
    lines.append(f"{s['n']} questions scored · scope `{run['scope']}` · extractor "
                 f"`{run['extractor']['model']}` · {len(payload['skipped'])} skipped "
                 f"(document not in store) · store `{run['store']}`")
    lines += ["", f"## {STRICT_LABEL}", "",
              "| accuracy | F1 | recall | precision |", "|---|---|---|---|",
              f"| **{s['accuracy']:.3f}** | **{s['f1']:.3f}** | {_fmt(s['recall'])} | {_fmt(s['precision'])} |",
              "", f"## {LENIENT_LABEL}", "",
              "| accuracy | F1 | recall | precision |", "|---|---|---|---|",
              f"| {l['accuracy']:.3f} | {l['f1']:.3f} | {_fmt(l['recall'])} | {_fmt(l['precision'])} |",
              "", "## Outcomes", "", "| outcome | count |", "|---|---|",
              f"| answered (fully verified) | {o['answered']} |",
              f"| partial (some claims unverified) | {o['partial']} |",
              f"| declined | {o['declined']} |", "",
              "| metric | value |", "|---|---|",
              f"| recall (answered in full) | {o['recall']:.1%} |",
              f"| recall incl. partial (upper bound) | {o['recall_incl_partial']:.1%} |",
              f"| false positive rate (any answer where a decline was expected) | {o['false_positive_rate']:.1%} |",
              f"| false positive rate, verified answers only | {o['false_positive_rate_verified_only']:.1%} |",
              f"| partial on should-decline (accepted tradeoff) | {o['partial_on_decline']} |",
              f"| partial on should-answer | {o['partial_on_answerable']} |",
              f"| refusal precision | {o['refusal_precision']:.1%} |"]
    for dim, rows in payload["breakdowns"].items():
        lines += ["", f"## By {dim}", "", "| slice | n | strict acc | lenient acc |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['n']} | {_fmt(v['strict_accuracy'])} | {_fmt(v['lenient_accuracy'])} |"
                  for k, v in rows.items()]
    lines += ["", "## Per question", "", "| id | stratum | want | outcome | strict | lenient |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['group']} | {r['expected_verdict']} | {r['outcome']} "
              f"| {r['score']:.2f} | {r['lenient_score']:.2f} |" for r in payload["questions"]]
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def _real_query_fn() -> Callable[[str], Any]:
    from verirag.generation.generator import AnswerFormat
    from verirag.graph.build import run_query

    return lambda q: run_query(q, AnswerFormat.CONCISE, corpus_only=True)


def run_benchmark(
    subset_path: str | Path,
    store: str | Path,
    *,
    extractor_kind: str = "deepseek",
    out_dir: str | Path = "eval_results",
    scope: str = "pooled",
    ids: list[int] | None = None,
    limit: int | None = None,
    only_ingested: bool = False,
    allow_missing: bool = False,
    resume: bool = True,
    ingested_docs: list[dict] | None = None,
    query_fn: Callable[[str], Any] | None = None,
    extract_fn: Callable[[str, str], str] | None = None,
    log: Callable[[str], None] = lambda m: print(m, flush=True),
) -> dict:
    """Run the benchmark sequentially and return (and write) the payload.

    ``ingested_docs``, ``query_fn`` and ``extract_fn`` are injection points for
    tests; left as ``None`` they use the store, the real pipeline and the real
    extractor.
    """
    if scope not in SCOPES:
        raise RunnerError(f"scope must be one of {SCOPES}")
    subset = json.loads(Path(subset_path).read_text())
    store_path = configure_store(store)
    if ingested_docs is None:
        from verirag.retrieval.vector_store import list_ingested_documents

        ingested_docs = list_ingested_documents()

    plan = make_plan(subset["questions"], ingested_docs, only_ingested=only_ingested,
                     allow_missing=allow_missing, ids=ids, limit=limit)
    log(f"store {store_path}: {len(plan.docs_present)}/{len(plan.docs_needed)} needed "
        f"documents present; {len(plan.todo)} questions to run, "
        f"{len(plan.skipped)} skipped (no document)")

    client = None
    model = "injected"
    if extract_fn is None:
        kind, model, client = build_extractor(extractor_kind)
        prompt = load_extraction_prompt()
        extract_fn = lambda q, t: extract_answer(q, t, prompt, client=client, model=model)  # noqa: E731
    query_fn = query_fn or _real_query_fn()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = subset.get("name", "mmlongbench")
    checkpoint = out / f"{name}_PARTIAL.json"
    done: list[dict] = []
    if resume and checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous.get("run", {}).get("scope") == scope:
            done = [r for r in previous.get("questions", []) if r["id"] in {q["id"] for q in plan.todo}]
            log(f"resuming: {len(done)} questions already done")
    finished = {r["id"] for r in done}

    def payload_for(records: list[dict], complete: bool) -> dict:
        return dict(
            name=name, date=date.today().isoformat(), complete=complete,
            run=dict(scope=scope, store=str(store_path), subset=str(subset_path),
                     extractor=dict(kind=extractor_kind, model=model,
                                    calls=getattr(client, "calls", None),
                                    errors=getattr(client, "errors", [])),
                     only_ingested=only_ingested),
            docs=dict(needed=plan.docs_needed, present=plan.docs_present,
                      missing=plan.docs_missing),
            skipped=[dict(id=q["id"], doc_id=q["doc_id"]) for q in plan.skipped],
            questions=records, **summarise(records),
        )

    records = list(done)
    todo = [q for q in plan.todo if q["id"] not in finished]
    for n, row in enumerate(todo, 1):
        record = process_question(row, query_fn=query_fn, extractor=extract_fn, scope=scope)
        records.append(record)
        checkpoint.write_text(json.dumps(payload_for(records, False), indent=2, default=str))
        flag = "" if record["score"] == record["lenient_score"] else f" (lenient {record['lenient_score']:.2f})"
        log(f"[{n}/{len(todo)}] Q{record['id']:>4} {record['group']:20} "
            f"{record['outcome']:9} strict={record['score']:.2f}{flag} ({record['elapsed_s']}s)")

    payload = payload_for(records, True)
    stem = f"{name}_{date.today().isoformat()}"
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str))
    (out / f"{stem}.md").write_text(render_report(payload))
    s, l = payload["metrics_strict"], payload["metrics_lenient"]
    log(f"\n{STRICT_LABEL}\n  accuracy={s['accuracy']:.3f} f1={s['f1']:.3f} "
        f"recall={_fmt(s['recall'])} precision={_fmt(s['precision'])} n={s['n']}")
    log(f"{LENIENT_LABEL}\n  accuracy={l['accuracy']:.3f} f1={l['f1']:.3f}")
    log(f"report: {out / (stem + '.md')}")
    return payload
