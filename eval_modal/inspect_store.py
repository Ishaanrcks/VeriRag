"""Read-only inspection of the ingested store: page coverage and gold-answer presence.

    modal run eval_modal/inspect_store.py

Answers two things the results JSON cannot:
  * stage (a) of the funnel -- is the gold answer present in any STORED chunk from a
    gold evidence page? (i.e. did ingestion capture the detail at all)
  * per-page coverage -- which pages have a text chunk, a table chunk, a caption, or
    nothing at all ("dark pages").

Never writes to the Volume: it only reloads and reads, so it is safe to run beside
a benchmark job.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal  # noqa: E402

from eval_modal.image import (  # noqa: E402
    APP_PREFIX, CPU, MEMORY_MIB, RESULTS, STORE, image, secret, volume, volumes,
)

app = modal.App(f"{APP_PREFIX}-inspect", image=image)


@app.function(cpu=CPU, memory=MEMORY_MIB, timeout=3600, retries=0,
              volumes=volumes, secrets=[secret()])
def inspect(results_file: str) -> dict:
    volume.reload()
    from verirag.eval.harness import configure_store
    from verirag.retrieval import vector_store

    configure_store(STORE)
    got = vector_store.get_collection().get(include=["metadatas", "documents"])
    chunks = list(zip(got["metadatas"], got["documents"]))

    # doc_id in the store is "<stem>-<hash>"; the benchmark's doc_id is the filename
    by_page: dict[tuple[str, int], list[tuple[str, str]]] = {}
    pages_seen: dict[str, set] = {}
    inventory: dict[str, dict] = {}
    for m, text in chunks:
        name = Path(str(m.get("source_path") or "")).name
        page = m.get("page")
        ct = str(m.get("content_type"))
        inv = inventory.setdefault(name, {"text": 0, "table": 0, "figure_caption": 0,
                                          "truncated": 0, "vector_render": 0})
        inv[ct] = inv.get(ct, 0) + 1
        if m.get("caption_truncated"):
            inv["truncated"] += 1
        if m.get("figure_source") == "vector_render":
            inv["vector_render"] += 1
        if isinstance(page, int):
            by_page.setdefault((name, page), []).append((ct, text or ""))
            pages_seen.setdefault(name, set()).add(page)

    results = json.loads((Path(RESULTS) / results_file).read_text())
    sq = lambda t: re.sub(r"[^0-9a-z.%]+", "", str(t).lower())  # noqa: E731

    def gold_items(q):
        g = q["ground_truth"]
        try:
            v = ast.literal_eval(g) if str(g).strip().startswith("[") else g
        except Exception:
            v = g
        return [str(i) for i in (v if isinstance(v, list) else [v]) if str(i).strip()]

    stage_a = {}
    for q in results["questions"]:
        if q["expected_verdict"] != "answer":
            continue
        items = [sq(i) for i in gold_items(q) if len(sq(i)) >= 2]
        found_in = []
        for page in q["evidence_pages"] or []:
            for ct, text in by_page.get((q["doc_id"], page), []):
                if items and all(i in sq(text) for i in items):
                    found_in.append({"page": page, "type": ct})
        stage_a[str(q["id"])] = dict(
            group=q["group"], present=bool(found_in), where=found_in[:3],
            pages_have_any_chunk={str(p): sorted({c for c, _ in by_page.get((q["doc_id"], p), [])})
                                  for p in (q["evidence_pages"] or [])},
        )
    return dict(
        store_chunks=len(chunks),
        inventory=inventory,
        pages_with_chunks={k: len(v) for k, v in pages_seen.items()},
        stage_a=stage_a,
    )


@app.local_entrypoint()
def main(results_file: str = "mmlongbench_subset200_v1canonical_tier2_2026-11-01.json"):
    out = inspect.remote(results_file)
    p = Path("/tmp/claude-501/store_inspect.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    a = out["stage_a"]
    print(f"store chunks: {out['store_chunks']}  documents: {len(out['inventory'])}")
    print(f"stage (a) gold answer present in a stored chunk from a gold page: "
          f"{sum(1 for v in a.values() if v['present'])}/{len(a)}")
    print(f"written to {p}")
