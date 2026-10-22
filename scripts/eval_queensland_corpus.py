"""QueensLand research-paper corpus evaluation.

Ingests every supported document from the QueensLand folder into an ISOLATED
eval store (./eval_store/queensland/), separate from the live store the
Streamlit app uses, then runs a fixed question set through run_query() with
corpus_only=True for every question.

The question set and its expected-answerable labels were written by hand after
reading the papers, so they are ground truth independent of the system under
test. Raw per-question traces are written as JSON for reuse by the Phase 12
harness, alongside a draft markdown table.

    python scripts/eval_queensland_corpus.py
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_STORE = PROJECT_ROOT / "eval_store" / "queensland"
RESULTS_DIR = PROJECT_ROOT / "eval_results"
CORPUS_DIR = Path("/Users/ishanyashpatel/Documents/QueensLand")

# Point every store at the isolated eval location BEFORE anything reads them.
from verirag.config import settings  # noqa: E402

EVAL_STORE.mkdir(parents=True, exist_ok=True)
settings.chroma_persist_dir = str(EVAL_STORE / "chroma")
settings.bm25_corpus_path = str(EVAL_STORE / "bm25_corpus.jsonl")
settings.metrics_db_path = str(EVAL_STORE / "metrics.db")

from verirag.generation.generator import AnswerFormat  # noqa: E402
from verirag.graph.build import run_query  # noqa: E402
from verirag.ingest import ingest_file  # noqa: E402
from verirag.loaders.base import EXTENSION_TO_FORMAT  # noqa: E402
from verirag.observability.metrics import summarize_metrics  # noqa: E402
from verirag.retrieval.vector_store import list_ingested_documents  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Question set. Labels are MY ground truth from reading the papers:
#   answerable  - the corpus genuinely contains the answer
#   unanswerable- the corpus does not support it (but the premise is fair)
#   false_premise- the question presupposes something the papers don't establish
# ---------------------------------------------------------------------------
QUESTIONS: list[dict] = [
    # ---- single-paper factual ----
    dict(id=1, kind="single", label="answerable",
         q="How many parameters does the GFM-RAG graph foundation model have?",
         truth="8M parameters."),
    dict(id=2, kind="single", label="answerable",
         q="How many knowledge graphs, triples and documents was GFM-RAG trained on?",
         truth="60 knowledge graphs, over 14M triples, 700k documents."),
    dict(id=3, kind="single", label="answerable",
         q="How many parameters does the G-reasoner graph foundation model have?",
         truth="34M parameters."),
    dict(id=4, kind="single", label="answerable",
         q="What is QuadGraph and how many layers does it have?",
         truth="A standardized four-layer abstraction unifying heterogeneous "
               "graph-structured knowledge into one representation."),
    dict(id=5, kind="single", label="answerable",
         q="Which four benchmark datasets is ExpressGNN evaluated on?",
         truth="UW-CSE, Cora, synthetic Kinship, and FB15K-237."),
    dict(id=6, kind="single", label="answerable",
         q="What two model families does ExpressGNN combine, and what does it "
           "use graph neural networks for?",
         truth="Markov Logic Networks and GNNs; GNNs perform variational "
               "inference in the MLN."),
    dict(id=7, kind="single", label="answerable",
         q="What coverage and interval width improvements does QpiGNN report "
           "over its baselines?",
         truth="On average 22% higher coverage and 50% narrower intervals."),
    dict(id=8, kind="single", label="answerable",
         q="What architecture does QpiGNN use to separate prediction from "
           "uncertainty estimation?",
         truth="A dual-head architecture that decouples prediction and "
               "uncertainty, trained with a quantile-free joint loss."),
    dict(id=9, kind="single", label="answerable",
         q="Which Qwen3 model sizes are used in the OPSD experiments?",
         truth="Qwen3-1.7B, Qwen3-4B and Qwen3-8B (instruct-tuned)."),
    dict(id=10, kind="single", label="answerable",
         q="In OPSD, what privileged information does the teacher policy "
           "condition on that the student does not see?",
         truth="Verified reasoning traces / ground-truth solutions; the "
               "student sees only the question."),
    dict(id=11, kind="single", label="answerable",
         q="What generation token caps are used for OPSD compared with GRPO?",
         truth="1024 tokens for OPSD, 16k for GRPO."),
    dict(id=12, kind="single", label="answerable",
         q="How many benchmark datasets is G-reasoner evaluated on?",
         truth="Six benchmark datasets."),
    # ---- genuine multi-paper synthesis ----
    dict(id=13, kind="synthesis", label="answerable",
         q="How does G-reasoner's parameter count compare with GFM-RAG's, and "
           "what limitation of GFM-RAG does G-reasoner set out to address?",
         truth="34M vs 8M. G-reasoner states GFM-RAG is limited to specific "
               "knowledge graphs and cannot generalize to other graph "
               "structures."),
    dict(id=14, kind="synthesis", label="answerable",
         q="Compare the graph representation used by GFM-RAG with the one used "
           "by G-reasoner.",
         truth="GFM-RAG reasons over a knowledge graph built from documents; "
               "G-reasoner unifies heterogeneous sources into QuadGraph, a "
               "standardized four-layer abstraction."),
    dict(id=15, kind="synthesis", label="answerable",
         q="Both ExpressGNN and GFM-RAG claim their models generalize to "
           "knowledge graphs not seen during training. How does each describe "
           "that capability?",
         truth="ExpressGNN: GNN embeddings may generalize to new entities or "
               "different but related KGs without retraining. GFM-RAG: first "
               "GFM applicable to unseen datasets without fine-tuning."),
    dict(id=16, kind="synthesis", label="answerable",
         q="Which papers in this corpus apply graph neural networks, and to "
           "what different problems does each apply them?",
         truth="GFM-RAG and G-reasoner (retrieval for RAG), ExpressGNN "
               "(probabilistic logic inference in MLNs), QpiGNN (uncertainty "
               "quantification for node regression)."),
    dict(id=17, kind="synthesis", label="answerable",
         q="Contrast how GFM-RAG and G-reasoner each scale training to large "
           "datasets.",
         truth="GFM-RAG: two-stage training over 60 KGs/14M triples/700k docs. "
               "G-reasoner: mixed-precision training and distributed "
               "message-passing across multiple GPUs."),
    dict(id=18, kind="synthesis", label="answerable",
         q="Do GFM-RAG and G-reasoner share any authors?",
         truth="Yes - Linhao Luo, Zicheng Zhao, Chen Gong, Gholamreza Haffari, "
               "Dinh Phung and Shirui Pan appear on both."),
    # ---- corpus-level question the system is not designed for ----
    dict(id=19, kind="synthesis", label="unanswerable",
         q="Do any two documents in this corpus describe the same paper, and "
           "if so which ones?",
         truth="GFM RAG.pdf and 2502.01113v3.pdf are the same paper in two "
               "versions, but answering needs corpus-level aggregation the "
               "system does not perform."),
    # ---- deliberate false premises ----
    dict(id=20, kind="false_premise", label="false_premise",
         q="What uncertainty quantification method do all of these papers use "
           "to produce prediction intervals?",
         truth="Only QpiGNN does uncertainty quantification; the others do "
               "not produce prediction intervals. Expect abstention."),
    dict(id=21, kind="false_premise", label="false_premise",
         q="How does OPSD's on-policy self-distillation improve GFM-RAG's "
           "graph retrieval performance?",
         truth="No such relationship exists; OPSD has no graph component and "
               "the papers are unrelated. Expect abstention."),
]


def _rule(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def _corpus_files() -> list[Path]:
    files = [
        path
        for path in sorted(CORPUS_DIR.rglob("*"))
        if path.is_file() and path.suffix.lower() in EXTENSION_TO_FORMAT
    ]
    return files


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"eval store : {EVAL_STORE}")
    print(f"corpus     : {CORPUS_DIR}")

    _rule("STEP 2 - ingest the corpus into the isolated eval store")
    files = _corpus_files()
    print(f"{len(files)} supported file(s) found")
    ingest_records = []
    for path in files:
        result = ingest_file(path)
        ingest_records.append(
            dict(
                file=str(path),
                doc_id=result.doc_id,
                text_chunks=result.num_text_chunks,
                figure_chunks=result.num_figure_chunks,
                skipped=result.skipped,
            )
        )
        print(f"  {path.name}: text={result.num_text_chunks} "
              f"fig={result.num_figure_chunks} skipped={result.skipped}")

    docs = list_ingested_documents()
    print(f"\ndistinct documents in eval store: {len(docs)}")

    _rule("STEP 4 - run every question (corpus_only=True)")
    records = []
    for item in QUESTIONS:
        print("-" * 78)
        print(f"[Q{item['id']}] ({item['kind']}, expected={item['label']})")
        print(f"  {item['q']}")
        result = run_query(item["q"], AnswerFormat.CONCISE, corpus_only=True)

        verdict = "abstained" if result.abstained else "answered"
        print(f"  -> {verdict} (regenerations={result.regeneration_count})")
        if result.abstained:
            print(f"     reason: {(result.abstention_reason or '')[:200]}")
        else:
            print(f"     answer: {(result.answer or '')[:300]}")
            for c in result.citations:
                print(f"       [{c.number}] {Path(str(c.source_path)).name}")

        records.append(
            dict(
                id=item["id"],
                kind=item["kind"],
                expected_label=item["label"],
                question=item["q"],
                ground_truth=item["truth"],
                verdict=verdict,
                answer=result.answer,
                abstention_reason=result.abstention_reason,
                regeneration_count=result.regeneration_count,
                citations=[dataclasses.asdict(c) | {"parent_text": None}
                           for c in result.citations],
                trace=result.trace,
            )
        )

    stamp = date.today().isoformat()
    traces_path = RESULTS_DIR / f"queensland_traces_{stamp}.json"
    summary = summarize_metrics()
    payload = dict(
        date=stamp,
        corpus_dir=str(CORPUS_DIR),
        eval_store=str(EVAL_STORE),
        ingested=ingest_records,
        documents=docs,
        questions=records,
        metrics=dataclasses.asdict(summary),
    )
    traces_path.write_text(json.dumps(payload, indent=2, default=str))

    _rule("SUMMARY")
    answered = [r for r in records if r["verdict"] == "answered"]
    abstained = [r for r in records if r["verdict"] == "abstained"]
    print(f"total questions   : {len(records)}")
    print(f"answered          : {len(answered)} "
          f"({len(answered) / len(records):.0%})")
    print(f"abstained         : {len(abstained)} "
          f"({len(abstained) / len(records):.0%})")
    print(f"avg regenerations : "
          f"{sum(r['regeneration_count'] for r in records) / len(records):.2f}")
    print(f"\nby expected label:")
    for label in ("answerable", "unanswerable", "false_premise"):
        subset = [r for r in records if r["expected_label"] == label]
        if not subset:
            continue
        got = len([r for r in subset if r["verdict"] == "answered"])
        print(f"  {label:14}: {len(subset)} questions, {got} answered, "
              f"{len(subset) - got} abstained")
    print(f"\nmetrics: {summary}")
    print(f"\ntraces written to {traces_path}")


if __name__ == "__main__":
    main()
