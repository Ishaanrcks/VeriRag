"""Small end-to-end pilot of the MMLongBench-Doc measurement chain.

    pipeline (retrieval -> Gate 1 -> generate/abstain -> Gate 2)
      -> LLM answer extraction -> vendored official scorer -> acc / F1

Validates the chain on a handful of questions whose PDFs are already in the
given store; the score itself is not meaningful at this size.

    python scripts/run_mmlongbench_pilot.py \
        --v1 <MMLongBench-Doc>/data/samples.json --store <store dir> \
        --docs a.pdf,b.pdf,c.pdf --out pilot.json [--extractor gemini|deepseek]

``--extractor deepseek`` (the default) uses the DeepSeek chat model;
``--extractor gemini`` uses settings.gemini_extraction_model over Gemini's
OpenAI-compatible endpoint and needs a billed GEMINI_API_KEY (the free tier
allows 20 requests a day). Extractor calls and failures are counted.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import random
import time
from pathlib import Path

from verirag.config import settings
from verirag.eval.harness import configure_store
from verirag.eval.mmlongbench_scorer import (
    FAILED_TO_EXTRACT,
    eval_acc_and_f1,
    extract_answer,
    load_extraction_prompt,
    make_extraction_client,
    parse_extraction,
    eval_score,
)


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


def pick_questions(rows: list[dict], docs: list[str], n_unans: int, n_ans: int, seed: int):
    """Deterministic mix: n_unans unanswerable + n_ans answerable, spread over formats."""
    rng = random.Random(seed)
    pool = [r for r in rows if r["doc_id"] in docs]
    unans = [r for r in pool if r["answer"].strip() == "Not answerable"]
    picked = rng.sample(unans, min(n_unans, len(unans)))
    by_format: dict[str, list[dict]] = collections.defaultdict(list)
    for r in pool:
        if r["answer"].strip() != "Not answerable":
            by_format[r["answer_format"]].append(r)
    formats = sorted(by_format)
    i = 0
    while len([p for p in picked if p["answer"].strip() != "Not answerable"]) < n_ans:
        fmt = formats[i % len(formats)]
        choices = [r for r in by_format[fmt] if r not in picked]
        if choices:
            picked.append(rng.choice(choices))
        i += 1
    return picked


def build_extractor(kind: str):
    if kind == "gemini":
        if not settings.gemini_api_key:
            raise SystemExit("GEMINI_API_KEY is not set")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--docs", required=True, help="comma-separated doc_id filenames")
    parser.add_argument("--out", required=True)
    parser.add_argument("--extractor", default="deepseek", choices=["deepseek", "gemini"])
    parser.add_argument("--unanswerable", type=int, default=3)
    parser.add_argument("--answerable", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_store(args.store)
    from verirag.generation.generator import AnswerFormat
    from verirag.graph.build import run_query

    rows = json.loads(Path(args.v1).read_text())
    questions = pick_questions(
        rows, args.docs.split(","), args.unanswerable, args.answerable, args.seed
    )
    kind, model, client = build_extractor(args.extractor)
    print(f"extractor: {kind} / {model}"
          + ("   (STAND-IN, not Gemini)" if kind == "deepseek" else ""),
          flush=True)
    prompt = load_extraction_prompt()

    samples = []
    for n, row in enumerate(questions, 1):
        started = time.perf_counter()
        result = run_query(row["question"], AnswerFormat.CONCISE, corpus_only=True)
        response = response_text(result)
        extracted = extract_answer(
            row["question"], response, prompt, client=client, model=model
        )
        pred = parse_extraction(extracted)
        score = eval_score(row["answer"], pred, row["answer_format"])
        g1 = result.gate1_verdict
        samples.append(dict(
            n=n, doc_id=row["doc_id"], question=row["question"], answer=row["answer"],
            answer_format=row["answer_format"], evidence_pages=row["evidence_pages"],
            abstained=result.abstained, abstention_reason=result.abstention_reason,
            system_answer=result.answer, response=response, extracted_res=extracted,
            pred=pred, score=score,
            gate1=(f"{g1.verdict}/{g1.decision_path}/{g1.max_rerank_score:.3f}" if g1 else None),
            regenerations=result.regeneration_count,
            # Full gate output and the structured per-stage log, for diagnosing
            # WHY a question was declined rather than only which gate declined.
            gate1_detail=dataclasses.asdict(g1) if g1 else None,
            gate2_detail=(dataclasses.asdict(result.gate2_verdict)
                          if result.gate2_verdict else None),
            pipeline_log=[dataclasses.asdict(e) for e in result.pipeline_log],
            elapsed_s=round(time.perf_counter() - started, 1),
            trace=result.trace,
        ))
        print(f"[{n}/{len(questions)}] score={score:.2f} gt={row['answer'][:30]!r} "
              f"pred={pred[:30]!r} abstained={result.abstained}", flush=True)

    acc, f1 = eval_acc_and_f1(samples)
    answerable = [s for s in samples if s["answer"] != "Not answerable"]
    predicted = [s for s in samples if s["pred"] != "Not answerable"]
    num = sum(s["score"] for s in answerable)
    payload = dict(
        extractor=dict(kind=kind, model=model, calls=client.calls, errors=client.errors),
        metrics=dict(
            n=len(samples), accuracy=acc, f1=f1,
            recall=num / len(answerable) if answerable else None,
            precision=num / len(predicted) if predicted else None,
            failed_to_extract=sum(1 for s in samples if s["pred"] == FAILED_TO_EXTRACT),
        ),
        samples=samples,
    )
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload["metrics"], indent=2))
    print(f"extractor calls: {client.calls}; errors: {len(client.errors)}")


if __name__ == "__main__":
    main()
