"""Small end-to-end pilot of the MMLongBench-Doc measurement chain.

    pipeline (retrieval -> Gate 1 -> generate/abstain -> Gate 2)
      -> LLM answer extraction -> vendored official scorer -> acc / F1

Validates the chain on a handful of questions whose PDFs are already in the
given store; the score itself is not meaningful at this size.

    python scripts/run_mmlongbench_pilot.py \
        --v1 <MMLongBench-Doc>/data/samples.json --store <store dir> \
        --docs a.pdf,b.pdf,c.pdf --out pilot.json [--extractor gemini|deepseek]

Every question is scored two ways: STRICT (the headline; for a partial answer the
extractor reads only its Verified facts) and LENIENT (the whole response; an
upper bound that can credit statements the system left unverified).

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

from verirag.eval.harness import configure_store
from verirag.eval.mmlongbench_runner import CountingClient, build_extractor, response_text
from verirag.eval.mmlongbench_scorer import (
    LENIENT_LABEL,
    STRICT_LABEL,
    extract_answer,
    load_extraction_prompt,
    pooled_metrics,
    score_response_dual,
)


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
        # Both passes: STRICT reads only the Verified facts (the headline),
        # LENIENT the whole response. They differ only for partial answers.
        dual = score_response_dual(
            row,
            response,
            lambda q, text: extract_answer(q, text, prompt, client=client, model=model),
        )
        strict, lenient = dual["strict"], dual["lenient"]
        pred, score = strict["pred"], strict["score"]
        extracted = strict["extracted_res"]
        g1 = result.gate1_verdict
        samples.append(dict(
            n=n, doc_id=row["doc_id"], question=row["question"], answer=row["answer"],
            answer_format=row["answer_format"], evidence_pages=row["evidence_pages"],
            abstained=result.abstained, abstention_reason=result.abstention_reason,
            outcome=result.outcome,
            verified_facts=[f.text for f in result.verified_facts],
            system_answer=result.answer, response=response, extracted_res=extracted,
            pred=pred, score=score,  # STRICT, the headline
            lenient_pred=lenient["pred"], lenient_score=lenient["score"],
            lenient_extracted_res=lenient["extracted_res"],
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
        flag = "" if lenient["score"] == score else f"   (lenient {lenient['score']:.2f})"
        print(f"[{n}/{len(questions)}] strict={score:.2f} gt={row['answer'][:30]!r} "
              f"pred={pred[:30]!r} outcome={result.outcome}{flag}", flush=True)

    strict_metrics = pooled_metrics(
        [dict(s, pred=s["pred"], score=s["score"]) for s in samples]
    )
    lenient_metrics = pooled_metrics(
        [dict(s, pred=s["lenient_pred"], score=s["lenient_score"]) for s in samples]
    )
    payload = dict(
        extractor=dict(kind=kind, model=model, calls=client.calls, errors=client.errors),
        metrics=strict_metrics,  # the headline is STRICT
        metrics_strict=strict_metrics,
        metrics_lenient=lenient_metrics,
        samples=samples,
    )
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    for label, m in ((STRICT_LABEL, strict_metrics), (LENIENT_LABEL, lenient_metrics)):
        print(f"\n{label}\n  accuracy={m['accuracy']:.3f}  f1={m['f1']:.3f}  "
              f"recall={m['recall']:.3f}  precision={m['precision']}  n={m['n']}")
    print(f"\nextractor calls: {client.calls}; errors: {len(client.errors)}")


if __name__ == "__main__":
    main()
