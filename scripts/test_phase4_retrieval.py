"""Phase 4 smoke test: hybrid retrieval (dense + sparse -> RRF -> rerank).

Ingests tests/fixtures/ into ISOLATED temp stores, then runs the retrieval
pipeline stage by stage for a few queries so the intermediate rankings are
visible: dense, sparse, fused (pre-cap), and final reranked. Each stage is
timed in milliseconds.

Models are warmed up before the timing loop so the reported latencies reflect
steady-state inference, not one-off model download/load.

    python scripts/test_phase4_retrieval.py
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path

# Redirect both stores to throwaway temp locations BEFORE anything reads them.
from verirag.config import settings  # noqa: E402

_TMP_DIR = Path(tempfile.mkdtemp(prefix="verirag_phase4_"))
settings.chroma_persist_dir = str(_TMP_DIR / "chroma")
settings.bm25_corpus_path = str(_TMP_DIR / "bm25_corpus.jsonl")

from verirag.embeddings.embedder import embed_query  # noqa: E402
from verirag.ingest import ingest_directory  # noqa: E402
from verirag.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from verirag.retrieval.hybrid import hybrid_retrieve  # noqa: E402
from verirag.retrieval.reranker import get_reranker, rerank  # noqa: E402
from verirag.retrieval.sparse_index import sparse_search  # noqa: E402
from verirag.retrieval.vector_store import dense_search  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

QUERIES = [
    "quarterly revenue chart",
    "two-gate verification critic",
    "weather forecast tomorrow",  # out-of-corpus; just observe (Gate 1 is Phase 6)
]


def _snippet(text: str, n: int = 100) -> str:
    return text[:n].replace("\n", " ")


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _run_query(query: str) -> None:
    print("=" * 76)
    print(f"QUERY: {query!r}")
    print("=" * 76)

    t = time.perf_counter()
    dense_hits = dense_search(query, settings.retrieval_dense_top_k)
    t_dense = _ms(t)

    t = time.perf_counter()
    sparse_hits = sparse_search(query, settings.retrieval_sparse_top_k)
    t_sparse = _ms(t)

    t = time.perf_counter()
    fused = reciprocal_rank_fusion(
        dense_hits, sparse_hits, settings.retrieval_rrf_k
    )
    t_fusion = _ms(t)

    capped = fused[: settings.retrieval_rerank_candidate_cap]

    t = time.perf_counter()
    final = rerank(query, capped, settings.retrieval_final_top_k)
    t_rerank = _ms(t)

    print(f"-- DENSE top-5 (of {len(dense_hits)}) --")
    for hit in dense_hits[:5]:
        print(f"  #{hit.dense_rank} dist={hit.dense_distance:.4f} {hit.chunk_id}")
        print(f"      {_snippet(hit.raw_text)!r}")

    print(f"-- SPARSE top-5 (of {len(sparse_hits)}) --")
    for hit in sparse_hits[:5]:
        print(f"  #{hit.sparse_rank} score={hit.sparse_score:.4f} {hit.chunk_id}")
        print(f"      {_snippet(hit.raw_text)!r}")

    print(f"-- FUSED after RRF (all {len(fused)}, before cap) --")
    for i, hit in enumerate(fused, start=1):
        print(
            f"  #{i} fused={hit.fused_score:.6f} {hit.chunk_id} "
            f"(dense_rank={hit.dense_rank}, sparse_rank={hit.sparse_rank})"
        )

    print(f"-- FINAL after rerank (top {settings.retrieval_final_top_k}) --")
    for i, hit in enumerate(final, start=1):
        fused_pos = next(
            (j for j, f in enumerate(fused, start=1) if f.chunk_id == hit.chunk_id),
            None,
        )
        print(
            f"  #{i} rerank={hit.rerank_score:.4f} {hit.chunk_id} "
            f"(was #{fused_pos} after fusion)"
        )
        print(f"      full raw_text: {hit.raw_text}")

    print("-- STAGE TIMINGS (ms) --")
    print(f"  dense_search : {t_dense:8.2f}")
    print(f"  sparse_search: {t_sparse:8.2f}")
    print(f"  fusion       : {t_fusion:8.2f}")
    print(f"  rerank       : {t_rerank:8.2f}  ({len(capped)} candidates)")
    print(f"  TOTAL        : {t_dense + t_sparse + t_fusion + t_rerank:8.2f}")

    # Sanity: the orchestrator should agree with the staged run above.
    orchestrated = [h.chunk_id for h in hybrid_retrieve(query)]
    print(
        f"  hybrid_retrieve() agrees: "
        f"{orchestrated == [h.chunk_id for h in final]}"
    )


def main() -> None:
    print(f"Isolated stores under: {_TMP_DIR}")

    print("=" * 76)
    print("INGEST tests/fixtures/")
    for result in ingest_directory(FIXTURES_DIR):
        print(f"  {result}")

    # Warm up both models so the timings below exclude load/download cost.
    print("Warming up embedder + reranker (excluded from timings)...")
    embed_query("warmup")
    get_reranker().predict([("warmup", "warmup passage")])
    print("Warm-up done.")

    for query in QUERIES:
        _run_query(query)

    print("=" * 76)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
