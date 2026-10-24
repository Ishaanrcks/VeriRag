"""Run a question set through the eval harness.

    # full set
    python scripts/run_eval.py --questions eval_data/queensland_stratified.json

    # a subset, diffed against a previous run
    python scripts/run_eval.py \
        --questions eval_data/queensland_subset14.json \
        --baseline eval_results/queensland_subset14_2026-10-24.json \
        --ids 1,2,3

    # validate the metric arithmetic without calling any model
    python scripts/run_eval.py --selftest
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from verirag.eval.harness import (
    compute_metrics,
    load_question_set,
    run_question_set,
    write_outputs,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_STORE = "eval_store/queensland"
DEFAULT_OUT = "eval_results"


def _selftest() -> int:
    """Deterministic check of the metric arithmetic -- no model calls.

    Worth having separately: an end-to-end run cannot validate the maths,
    because the numbers it produces are exactly what is in question.
    """
    records = [
        # want answer, answered, correct
        dict(id=1, group="factoid", expected_verdict="answer",
             outcome="answered", correct=True, regeneration_count=0),
        # want answer, answered, WRONG -> a hallucination
        dict(id=2, group="factoid", expected_verdict="answer",
             outcome="answered", correct=False, regeneration_count=1),
        # want answer, abstained -> a false negative, hurts recall
        dict(id=3, group="multi_hop", expected_verdict="answer",
             outcome="abstained", correct=None, regeneration_count=2),
        # want decline, declined -> correct refusal
        dict(id=4, group="unanswerable", expected_verdict="decline",
             outcome="abstained", correct=None, regeneration_count=0),
        # want decline, ANSWERED -> a false positive
        dict(id=5, group="false_premise", expected_verdict="decline",
             outcome="answered", correct=None, regeneration_count=1),
    ]
    m = compute_metrics(records)
    checks = {
        "total == 5": m.total == 5,
        "answered == 3": m.answered == 3,
        "declined == 2": m.declined == 2,
        "recall == 2/3": abs(m.recall - 2 / 3) < 1e-9,
        "refusal_precision == 1/2": abs(m.refusal_precision - 0.5) < 1e-9,
        "false_positive_rate == 1/2": abs(m.false_positive_rate - 0.5) < 1e-9,
        "hallucination_rate == 1/2": abs(m.hallucination_rate - 0.5) < 1e-9,
        "scored_answers == 2": m.scored_answers == 2,
        "hallucinations == [2]": m.hallucinations == [2],
        "avg_regenerations == 0.8": abs(m.avg_regenerations - 0.8) < 1e-9,
        "strata counted": set(m.by_group) == {
            "factoid", "multi_hop", "unanswerable", "false_premise"},
        "factoid wrong == 1": m.by_group["factoid"]["wrong"] == 1,
    }
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    failed = [n for n, p in checks.items() if not p]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} metric checks passed")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", help="question-set JSON")
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--baseline", help="previous run's JSON, for a diff")
    parser.add_argument("--ids", help="comma-separated question ids to run")
    parser.add_argument("--selftest", action="store_true",
                        help="validate metric arithmetic and exit")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()
    if not args.questions:
        parser.error("--questions is required unless --selftest is given")

    question_set = load_question_set(args.questions)
    only_ids = (
        [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None
    )
    baseline = (
        json.loads(Path(args.baseline).read_text()) if args.baseline else None
    )

    checkpoint = (
        Path(args.out_dir)
        / f"{question_set.get('name', 'eval')}_PARTIAL.json"
    )
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"checkpointing after each question to {checkpoint}")
    payload = run_question_set(
        question_set, args.store, only_ids=only_ids, checkpoint_path=checkpoint
    )
    traces, report = write_outputs(payload, args.out_dir, baseline)

    m = payload["metrics"]
    print(f"\nrecall={m['recall']:.1%}  refusal_precision="
          f"{m['refusal_precision']:.1%}  fpr={m['false_positive_rate']:.1%}  "
          f"hallucination_rate={m['hallucination_rate']:.1%}")
    if m["hallucination_rate"] > 0:
        print(f"*** HALLUCINATION RATE NOT ZERO: questions "
              f"{m['hallucinations']} ***")
    print(f"traces: {traces}\nreport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
