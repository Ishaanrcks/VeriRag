"""Run one benchmark configuration as N parallel shards and merge the results.

    modal run --detach eval_modal/run_matrix.py --config M1a

A configuration is a name, a set of experiment flags, a retrieval scope and
whether the gates are on. ``CONFIGS`` holds the matrix; each one runs the full
question set split across ``SHARDS`` containers via ``.starmap``, and the shard
payloads are merged into a single report whose metrics are recomputed over the
union (never averaged across shards).

Results are returned to the caller and written to the CONTAINER's /tmp, never to
the Volume: several configurations run in parallel, and concurrent commits to one
Volume can clobber each other.
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

SUBSET = "eval_data/mmlongbench_subset200_v1canonical.json"
SHARDS = 4

# name -> (flags, scope, gates)
#
# NOTE after the final-c2 freeze: apply_flags declares a run's EXACT flag set, so a
# production default is NOT implicit. gate2_visual_verify is now on in production, so
# only the configs that name it (M1b, W1, W1-no-k, W1-no-fmt, C1, C2, P3, PC) reflect
# today's pipeline. M0, W0, R2 and the LOO runs are PRE-FREEZE baselines kept for
# reproducing the measurements; they no longer represent production.
CONFIGS: dict[str, tuple[str, str, bool]] = {
    # baseline: every flag off, the pipeline as it ships
    "M0": ("", "pooled", True),
    # batch 1 without the visual check, and with it
    "M1a": ("gate1_advisory,gate2_citation_repair,gate2_numeric_match,refusal_requires_search", "pooled", True),
    "M1b": ("gate1_advisory,gate2_citation_repair,gate2_numeric_match,refusal_requires_search,gate2_visual_verify", "pooled", True),
    # leave-one-out on M1a, to attribute its effect to individual flags
    "LOO-no-gate1": ("gate2_citation_repair,gate2_numeric_match,refusal_requires_search", "pooled", True),
    "LOO-no-repair": ("gate1_advisory,gate2_numeric_match,refusal_requires_search", "pooled", True),
    "LOO-no-numeric": ("gate1_advisory,gate2_citation_repair,refusal_requires_search", "pooled", True),
    "LOO-no-refusal": ("gate1_advisory,gate2_citation_repair,gate2_numeric_match", "pooled", True),
    # reference points: no gates at all
    "R2": ("", "pooled", False),
    # W0: M0 repeated, to measure run-to-run noise before crediting any change
    "W0": ("", "pooled", True),
    # W1: the three kept changes (c1 is an always-on bug fix, not a flag)
    "W1": ("retrieval_wide,extractor_format_hint,gate2_visual_verify", "pooled", True),
    # leave-one-out on W1, run only if W1 fails the bar
    "W1-no-k": ("extractor_format_hint,gate2_visual_verify", "pooled", True),
    "W1-no-fmt": ("retrieval_wide,gate2_visual_verify", "pooled", True),
    "W1-no-vis": ("retrieval_wide,extractor_format_hint", "pooled", True),
    # The kept changes only: c1 is an always-on bug fix, so the flag is just f_vis.
    # retrieval_wide is excluded -- it caused the Q18 and Q123 losses.
    "C1": ("gate2_visual_verify", "pooled", True),
    # C1 with the regeneration loop capped at 1 instead of 2.
    "C2": ("gate2_visual_verify,critic_iterations_one", "pooled", True),
    # P3 = C2 + p3_bundle (caption_slots + list_row_scope; fullest_form was dropped).
    "P3": ("gate2_visual_verify,critic_iterations_one,p3_bundle", "pooled", True),
    # PC = C2 + unverified_candidate. The headline score must equal C2's exactly:
    # the candidate is recorded separately and never scored.
    "PC": ("gate2_visual_verify,critic_iterations_one,unverified_candidate", "pooled", True),
}

app = modal.App(f"{APP_PREFIX}-matrix", image=image)


@app.function(cpu=CPU, memory=MEMORY_MIB, timeout=TIMEOUT_SECONDS, retries=0,
              volumes=volumes, secrets=[secret()])
def run_shard(config: str, index: int) -> dict:
    """One shard of one configuration. Reads the store; never commits the Volume."""
    volume.reload()
    from verirag.eval.flags import apply_flags
    from verirag.eval.mmlongbench_runner import run_benchmark

    flags, scope, gates = CONFIGS[config]
    apply_flags(flags)  # also pins generation temperature to 0
    return run_benchmark(
        SUBSET, STORE, out_dir=f"/tmp/matrix/{config}", scope=scope, gates=gates,
        shard=(index, SHARDS), only_ingested=True,
        # resume=True so a shard restarted mid-run (Modal can reclaim a container)
        # continues from its own checkpoint instead of redoing its 22 questions. W1
        # spent about a quarter of its runtime re-running 31 questions that way. It
        # cannot change results: the checkpoint is keyed on this config's tag and
        # shard, and only completed records are reused.
        resume=True, tag=config,
    )


@app.function(cpu=2, memory=4096, timeout=1800, retries=0, volumes=volumes, secrets=[secret()])
def merge_and_summarise(config: str, payloads: list[dict]) -> dict:
    """Merge the shard payloads and build the summary, INSIDE the image.

    This runs remotely on purpose: ``verirag`` only exists in the Modal image, not
    on the launching machine, and an earlier version did the merge in the local
    entrypoint and lost four completed runs to ModuleNotFoundError.
    """
    import tempfile

    from verirag.eval.mmlongbench_runner import merge_shards

    with tempfile.TemporaryDirectory() as tmp:
        merged = merge_shards(payloads, name=f"tier2_{config}", out_dir=tmp)
    rows = merged["questions"]
    answerable = [r for r in rows if r["expected_verdict"] == "answer"]
    decline = [r for r in rows if r["expected_verdict"] == "decline"]
    q165 = next((r for r in rows if r["id"] == 165), None)
    return dict(
        config=config, flags=CONFIGS[config][0], scope=CONFIGS[config][1],
        gates=CONFIGS[config][2], n=len(rows),
        strict_accuracy=merged["metrics_strict"]["accuracy"],
        lenient_accuracy=merged["metrics_lenient"]["accuracy"],
        answerable_correct=sum(1 for r in answerable if r["score"] > 0),
        answerable_n=len(answerable),
        unanswerable_correct=sum(1 for r in decline if r["score"] == 1.0),
        unanswerable_n=len(decline),
        unanswerable_lost=[r["id"] for r in decline if r["score"] < 1.0],
        q165=dict(score=q165["score"], pred=q165["pred"], outcome=q165["outcome"],
                  answer=(q165.get("answer") or "")[:200]) if q165 else None,
        outcomes={k: sum(1 for r in rows if r["outcome"] == k) for k in {r["outcome"] for r in rows}},
        visual_checks=sum(len(r.get("visual_checks") or []) for r in rows),
        citation_repairs=sum(len(r.get("citation_repairs") or []) for r in rows),
        per_question={str(r["id"]): dict(group=r["group"], score=r["score"], outcome=r["outcome"],
                                         pred=r["pred"][:40], expected=r["expected_verdict"],
                                         gate1=r.get("gate1"), gate2=r.get("gate2_labels"))
                      for r in rows},
    )


@app.local_entrypoint()
def main(config: str):
    """Fan out the shards, then merge REMOTELY.

    Nothing here imports verirag: this process only has modal. The raw shard
    payloads are written to disk the moment they arrive, so a failure in the merge
    can never again discard completed runs.
    """
    if config not in CONFIGS:
        raise SystemExit(f"unknown config {config!r}; known: {list(CONFIGS)}")
    out_dir = Path("/tmp/claude-501/matrix")
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = list(run_shard.starmap([(config, i) for i in range(1, SHARDS + 1)]))
    raw = out_dir / f"payloads_{config}.json"
    raw.write_text(json.dumps(payloads))          # durable before anything can fail
    print(f"{config}: {len(payloads)} shard payloads saved to {raw}")

    summary = merge_and_summarise.remote(config, payloads)
    (out_dir / f"summary_{config}.json").write_text(json.dumps(summary, indent=2))
    print(f"{config}: strict={summary['strict_accuracy']:.3f} "
          f"answerable {summary['answerable_correct']}/{summary['answerable_n']} "
          f"unanswerable {summary['unanswerable_correct']}/{summary['unanswerable_n']} "
          f"lost={summary['unanswerable_lost']}")
    print(f"Q165: {summary['q165']}")
