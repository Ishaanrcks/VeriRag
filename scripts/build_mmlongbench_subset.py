"""Build the stratified 200-question MMLongBench-Doc subset from canonical V1.

    python scripts/build_mmlongbench_subset.py \
        --v1 <MMLongBench-Doc>/data/samples.json \
        --v2-dir <MMLongBench-Doc-V2> \
        --out eval_data/mmlongbench_subset200_v1canonical.json

Canonical V1 is upstream GitHub's ``data/samples.json`` (1082 rows). The
HuggingFace parquet (1091 rows) is an older snapshot and is NOT used.

Stratification: proportional allocation (largest remainder) over the joint
``group x cross_page`` strata, seed 42.

  group      "unanswerable" if answer == "Not answerable", else the single
             evidence type (text/table/chart/layout/figure), "multiple" if
             more than one, "unlabeled_evidence" if answerable with an empty
             evidence_sources list.
  cross_page len(evidence_pages) > 1

Each question is also tagged with its status in MMLongBench-Doc-V2
(``v2_status``), joined on (doc_id, whitespace-normalised question):

  unchanged  identical in V2
  corrected  answer / evidence corrected, question text unchanged
  reworded   V2 changed the question text (the correction entry's
             ``manual_question``)
  removed    absent from V2: a ``corpus_defect`` drop, or the duplicate
             row V2 deletes
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

LABELS = {
    "Pure-text (Plain-text)": "text",
    "Table": "table",
    "Chart": "chart",
    "Generalized-text (Layout)": "layout",
    "Figure": "figure",
}


def norm(text: str) -> str:
    return " ".join(str(text).split())


def v2_tags(v1_rows: list[dict], v2_dir: Path) -> list[dict]:
    """Per-row V2 status, aligned with ``v1_rows``."""
    v2_rows = json.loads((v2_dir / "data" / "samples.json").read_text())
    corrections = json.loads(
        (v2_dir / "data" / "corrections" / "corrections.json").read_text()
    )
    v2_keys = {(r["doc_id"], norm(r["question"])) for r in v2_rows}
    # V1 holds the same (doc_id, question) twice with contradictory answers, so
    # the answer disambiguates which copy a correction refers to.
    by_key = {
        (c["doc_id"], norm(c["question"]), norm(c["benchmark_answer"])): c
        for c in corrections
    }
    v1_count = collections.Counter((r["doc_id"], norm(r["question"])) for r in v1_rows)

    tags = []
    for row in v1_rows:
        key = (row["doc_id"], norm(row["question"]))
        corr = by_key.get(key + (norm(row["answer"]),))
        tag: dict = {"v2_status": "unchanged", "v2_correction_reason": None}
        if corr is not None:
            reason = f"{corr['dispute_type']}: {corr.get('note', '')}".strip()
            tag["v2_correction_reason"] = reason
            if corr["dispute_type"] == "corpus_defect":
                tag["v2_status"] = "removed"
            elif corr.get("manual_question") and norm(corr["manual_question"]) != norm(
                corr["question"]
            ):
                tag["v2_status"] = "reworded"
                tag["v2_question"] = corr["manual_question"]
            else:
                tag["v2_status"] = "corrected"
            if tag["v2_status"] != "removed":
                tag["v2_ground_truth"] = corr["manual_answer"]
        elif v1_count[key] > 1:
            tag["v2_status"] = "removed"
            tag["v2_correction_reason"] = (
                "duplicate_row: V2 keeps one copy of this contradictory pair"
            )
        elif key not in v2_keys:
            tag["v2_status"] = "removed"
            tag["v2_correction_reason"] = "absent from V2 (no correction entry)"
        tags.append(tag)
    return tags


def record(index: int, row: dict, tag: dict) -> dict:
    import ast

    evidence = [LABELS[x] for x in ast.literal_eval(row["evidence_sources"])]
    pages = ast.literal_eval(row["evidence_pages"])
    unanswerable = row["answer"].strip() == "Not answerable"
    if unanswerable:
        group = "unanswerable"
    elif not evidence:
        group = "unlabeled_evidence"
    else:
        group = evidence[0] if len(evidence) == 1 else "multiple"
    return dict(
        source_index=index,
        group=group,
        question=row["question"],
        ground_truth=row["answer"],
        expected_verdict="decline" if unanswerable else "answer",
        doc_id=row["doc_id"],
        doc_type=row["doc_type"],
        evidence_pages=pages,
        evidence_sources=evidence,
        answer_format=row["answer_format"],
        cross_page=len(pages) > 1,
        **tag,
    )


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    strata: dict = collections.defaultdict(list)
    for r in records:
        strata[(r["group"], r["cross_page"])].append(r)
    quota = {k: len(v) * n / len(records) for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    leftovers = sorted(quota, key=lambda k: (quota[k] - alloc[k], str(k)), reverse=True)
    for k in leftovers[: n - sum(alloc.values())]:
        alloc[k] += 1
    rng = random.Random(seed)
    picked: list[dict] = []
    for k in sorted(strata, key=str):
        picked += rng.sample(strata[k], alloc[k])
    rng.shuffle(picked)
    return picked


def pct(counter: collections.Counter, total: int) -> dict:
    return {k: f"{v} ({v / total:.1%})" for k, v in sorted(counter.items())}


def summarise(records: list[dict]) -> dict:
    n = len(records)
    return dict(
        group=pct(collections.Counter(r["group"] for r in records), n),
        answerability=pct(collections.Counter(r["expected_verdict"] for r in records), n),
        cross_page=pct(
            collections.Counter("cross_page" if r["cross_page"] else "single_or_none"
                                for r in records), n),
        v2_status=pct(collections.Counter(r["v2_status"] for r in records), n),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", required=True, help="upstream data/samples.json")
    parser.add_argument("--v2-dir", required=True, help="MMLongBench-Doc-V2 checkout")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    v1_rows = json.loads(Path(args.v1).read_text())
    tags = v2_tags(v1_rows, Path(args.v2_dir))
    records = [record(i, r, t) for i, (r, t) in enumerate(zip(v1_rows, tags))]
    picked = stratified_sample(records, args.n, args.seed)
    for i, r in enumerate(picked, 1):
        r["id"] = i
    questions = [{"id": r["id"], **{k: v for k, v in r.items() if k != "id"}} for r in picked]
    docs = sorted({r["doc_id"] for r in questions})
    payload = dict(
        name="mmlongbench_subset200_v1canonical",
        source="https://github.com/mayubo2333/MMLongBench-Doc (data/samples.json, "
               f"{len(v1_rows)} questions)",
        paper="https://arxiv.org/abs/2407.01523",
        v2_source="https://github.com/VectifyAI/MMLongBench-Doc-V2",
        corpus=f"documents in the upstream repo's data/documents/; {len(docs)} PDFs "
               "needed, NOT yet downloaded",
        scope_note=(
            f"Stratified random sample of {args.n} of {len(v1_rows)}, seed {args.seed}, "
            "proportional allocation (largest remainder) over group x cross_page. "
            "group = 'unanswerable' if answer == 'Not answerable', else the single "
            "evidence type (text/table/chart/layout/figure), 'multiple' if >1, "
            "'unlabeled_evidence' if answerable with no evidence label. cross_page = "
            "len(evidence_pages) > 1. ground_truth is the canonical V1 key; where "
            "v2_status is 'corrected' or 'reworded', v2_ground_truth (and v2_question) "
            "hold V2's version. evidence_pages are 1-indexed. No must_contain: answers "
            "are typed and scored by the vendored official scorer."
        ),
        strata_counts=collections.Counter(r["group"] for r in questions),
        full_dataset_breakdown=summarise(records),
        subset_breakdown=summarise(questions),
        doc_ids=docs,
        questions=questions,
    )
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"v1 rows {len(v1_rows)}; subset {len(questions)}; docs {len(docs)}")
    full, sub = payload["full_dataset_breakdown"], payload["subset_breakdown"]
    for facet in full:
        print(f"== {facet}")
        for k in full[facet]:
            print(f"  {k:22} full {full[facet][k]:14} subset {sub[facet].get(k, '0')}")


if __name__ == "__main__":
    main()
