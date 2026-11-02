"""Run the MMLongBench-Doc question set end to end, sequentially.

    # what would run? (preflight only: no model is loaded, no question is asked)
    python scripts/run_mmlongbench_benchmark.py --store <store dir> --dry-run

    # the questions whose PDF is already ingested (how tiered ingestion is measured)
    python scripts/run_mmlongbench_benchmark.py --store <store dir> --only-ingested

    # everything; refuses to start if any needed document is not in the store
    python scripts/run_mmlongbench_benchmark.py --store <store dir>

Writes ``<out-dir>/<name>_<date>.json`` and ``.md`` and checkpoints to
``<name>_PARTIAL.json`` after every question; re-running resumes. STRICT is the
headline score; LENIENT is an upper bound that includes unverified statements.

Run ONE of these at a time: it loads the embedding, reranker and NLI models, and
two model-loading processes at once wedge an 8 GB machine. It checks available
memory before loading anything and before every question, and stops at its
checkpoint (exit code 3) rather than stall; ``--dry-run`` shows the current reading.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from verirag.eval.harness import configure_store
from verirag.eval.mmlongbench_runner import (
    DEFAULT_SUBSET,
    MAX_SECONDS_PER_QUESTION,
    MIN_AVAILABLE_GB,
    MIN_AVAILABLE_GB_DURING,
    MemoryLow,
    RunnerError,
    StallDetected,
    make_plan,
    memory_summary,
    run_benchmark,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--store", required=True, help="ingested store directory")
    parser.add_argument("--out-dir", default="eval_results")
    parser.add_argument("--extractor", default="deepseek", choices=["deepseek", "gemini"])
    parser.add_argument("--scope", default="pooled", choices=["pooled", "doc"],
                        help="pooled: search the whole store; doc: restrict to the question's file")
    parser.add_argument("--ids", help="comma-separated question ids")
    parser.add_argument("--limit", type=int, help="run at most N questions")
    parser.add_argument("--only-ingested", action="store_true",
                        help="run only questions whose document is in the store")
    parser.add_argument("--allow-missing", action="store_true",
                        help="record questions with no document as skipped instead of refusing")
    parser.add_argument("--no-gates", action="store_true",
                        help="ABLATION ONLY: bypass Gate 1 and Gate 2 (retrieve -> generate -> "
                             "extract). Production defaults are unchanged.")
    parser.add_argument("--tag", default="",
                        help="appended to the output names (e.g. tier1) so tiers do not overwrite each other")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="preflight only: coverage and current memory; loads no model")
    parser.add_argument("--min-available-gb", type=float, default=MIN_AVAILABLE_GB,
                        help="refuse to start with less available memory than this (0 disables)")
    parser.add_argument("--min-available-gb-during", type=float, default=MIN_AVAILABLE_GB_DURING,
                        help="stop at the checkpoint if available memory falls below this")
    parser.add_argument("--max-seconds-per-question", type=float, default=MAX_SECONDS_PER_QUESTION,
                        help="stop at the checkpoint if one question is slower (0 disables)")
    args = parser.parse_args()
    ids = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None

    try:
        if args.dry_run:
            from verirag.retrieval.vector_store import list_ingested_documents

            configure_store(args.store)
            subset = json.loads(Path(args.subset).read_text())
            plan = make_plan(subset["questions"], list_ingested_documents(), only_ingested=True,
                             ids=ids, limit=args.limit)
            print(f"subset {args.subset}: {len(subset['questions'])} questions, "
                  f"{len(plan.docs_needed)} documents")
            print(f"store {args.store}: {len(plan.docs_present)} of those documents present")
            print(f"runnable now: {len(plan.todo)} questions; "
                  f"blocked on a missing document: {len(plan.skipped)}")
            if plan.docs_missing:
                print(f"missing documents ({len(plan.docs_missing)}), first 5: "
                      f"{plan.docs_missing[:5]}")
            print(memory_summary(args.min_available_gb)[0])
            return 0
        run_benchmark(
            args.subset, args.store, extractor_kind=args.extractor, out_dir=args.out_dir,
            scope=args.scope, gates=not args.no_gates, ids=ids, limit=args.limit,
            only_ingested=args.only_ingested,
            allow_missing=args.allow_missing, resume=not args.no_resume, tag=args.tag,
            min_available_gb=args.min_available_gb,
            min_available_gb_during=args.min_available_gb_during,
            max_seconds_per_question=args.max_seconds_per_question,
        )
    except (MemoryLow, StallDetected) as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 3
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
