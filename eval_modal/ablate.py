"""Ablation: the same questions under four retrieval/gating configurations.

    modal run --detach eval_modal/ablate.py --config pooled-gates --ids 1,2,4

Answers one question: of the answerable failures, how many are caused by pooled
retrieval (the context is half other documents) and how many by the gates
declining or stripping a verified claim?

    pooled-gates   scope=pooled, gates on   (what production does today)
    doc-gates      scope=doc,    gates on   (oracle: retrieval filtered to the
                                             question's own document)
    pooled-nogates scope=pooled, gates off  (retrieve -> generate -> extract)
    doc-nogates    scope=doc,    gates off

Results are returned to the caller and written to the CONTAINER's /tmp, never to
the Volume: several of these run in parallel alongside a real benchmark job, and
concurrent commits to one Volume can clobber each other.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal  # noqa: E402

from eval_modal.image import (  # noqa: E402
    APP_PREFIX, CPU, MEMORY_MIB, STORE, TIMEOUT_SECONDS, image, secret, volume, volumes,
)

CONFIGS = {
    "pooled-gates": ("pooled", True),
    "doc-gates": ("doc", True),
    "pooled-nogates": ("pooled", False),
    "doc-nogates": ("doc", False),
}
SUBSET = "eval_data/mmlongbench_subset200_v1canonical.json"

app = modal.App(f"{APP_PREFIX}-ablate", image=image)


@app.function(cpu=CPU, memory=MEMORY_MIB, timeout=TIMEOUT_SECONDS, retries=0,
              volumes=volumes, secrets=[secret()])
def ablate(config: str, ids: str) -> dict:
    volume.reload()  # read the store the ingest job committed; never commit back
    from verirag.eval.mmlongbench_runner import run_benchmark

    scope, gates = CONFIGS[config]
    payload = run_benchmark(
        SUBSET, STORE, out_dir=f"/tmp/ablation/{config}", scope=scope, gates=gates,
        ids=[int(x) for x in ids.split(",") if x.strip()],
        only_ingested=True, resume=False, tag=config,
    )
    rows = payload["questions"]
    answerable = [r for r in rows if r["expected_verdict"] == "answer"]
    decline = [r for r in rows if r["expected_verdict"] == "decline"]
    return dict(
        config=config, scope=scope, gates=gates, n=len(rows),
        strict_accuracy=payload["metrics_strict"]["accuracy"],
        lenient_accuracy=payload["metrics_lenient"]["accuracy"],
        answerable_n=len(answerable),
        answerable_correct=sum(1 for r in answerable if r["score"] > 0),
        unanswerable_n=len(decline),
        unanswerable_correct=sum(1 for r in decline if r["score"] == 1.0),
        outcomes={k: sum(1 for r in rows if r["outcome"] == k)
                  for k in {r["outcome"] for r in rows}},
        per_question={str(r["id"]): dict(group=r["group"], score=r["score"],
                                         outcome=r["outcome"], pred=r["pred"][:40])
                      for r in rows},
    )


@app.local_entrypoint()
def main(config: str, ids: str):
    result = ablate.remote(config, ids)
    out = Path(f"/tmp/claude-501/ablation_{config}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"{config}: strict={result['strict_accuracy']:.3f} "
          f"answerable {result['answerable_correct']}/{result['answerable_n']} "
          f"unanswerable {result['unanswerable_correct']}/{result['unanswerable_n']}")
    print(f"written to {out}")
