"""Read-only: stored text for gold pages, and gold-page recall at several k.

    modal run eval_modal/probe_retrieval.py --ids 1,2,4

Two things the results JSON cannot answer:
  * what the store actually holds for a question's gold evidence pages (the chunk
    text, so it can be read beside the page's raw PyMuPDF text);
  * gold-page recall at k = 8, 20, 40 -- one retrieval per question at the largest
    k, then the prefix of the ranking is measured at each smaller k.

Embedder and reranker only, no LLM calls, and the Volume is never committed, so
this is safe to run beside live benchmark jobs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal  # noqa: E402

from eval_modal.image import (  # noqa: E402
    APP_PREFIX, CPU, MEMORY_MIB, STORE, image, secret, volume, volumes,
)

SUBSET = "eval_data/mmlongbench_subset200_v1canonical.json"
KS = (8, 20, 40)

app = modal.App(f"{APP_PREFIX}-probe-retrieval", image=image)


@app.function(cpu=CPU, memory=MEMORY_MIB, timeout=5400, retries=0,
              volumes=volumes, secrets=[secret()])
def probe(ids: str) -> dict:
    volume.reload()
    from verirag.config import settings
    from verirag.eval.harness import configure_store
    from verirag.retrieval import vector_store
    from verirag.retrieval.hybrid import hybrid_retrieve

    configure_store(STORE)
    wanted = {int(x) for x in ids.split(",") if x.strip()}
    questions = [q for q in json.loads(Path(SUBSET).read_text())["questions"]
                 if q["id"] in wanted]

    # stored chunks, indexed by (document filename, page)
    got = vector_store.get_collection().get(include=["metadatas", "documents"])
    by_page: dict[tuple[str, int], list[dict]] = {}
    for meta, text in zip(got["metadatas"], got["documents"]):
        page = meta.get("page")
        if isinstance(page, int):
            key = (Path(str(meta.get("source_path") or "")).name, page)
            by_page.setdefault(key, []).append(
                {"type": str(meta.get("content_type")), "text": text or ""}
            )

    # one retrieval per question at the largest k; smaller k are prefixes of it
    settings.retrieval_final_top_k = max(KS)
    settings.retrieval_rerank_candidate_cap = max(KS)
    settings.retrieval_dense_top_k = max(KS) * 2
    settings.retrieval_sparse_top_k = max(KS) * 2

    out = {}
    for q in questions:
        hits = hybrid_retrieve(q["question"], corpus_only=True)
        ranking = [
            {"doc": Path(str(h.metadata.get("source_path") or "")).name,
             "page": h.metadata.get("page"), "type": str(h.metadata.get("content_type")),
             "score": h.rerank_score}
            for h in hits
        ]
        gold = {(q["doc_id"], p) for p in (q["evidence_pages"] or [])}
        out[str(q["id"])] = {
            "recall_at": {str(k): any((r["doc"], r["page"]) in gold for r in ranking[:k])
                          for k in KS},
            "best_gold_rank": next((i + 1 for i, r in enumerate(ranking)
                                    if (r["doc"], r["page"]) in gold), None),
            "n_retrieved": len(ranking),
            "stored_gold_pages": {
                str(p): by_page.get((q["doc_id"], p), []) for p in (q["evidence_pages"] or [])
            },
        }
    return out


@app.local_entrypoint()
def main(ids: str):
    out = probe.remote(ids)
    p = Path("/tmp/claude-501/retrieval_probe.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    for k in KS:
        hits = sum(1 for v in out.values() if v["recall_at"][str(k)])
        print(f"  gold-page recall @k={k:<3} {hits}/{len(out)}")
    print(f"written to {p}")
