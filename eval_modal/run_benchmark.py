"""Run the MMLongBench-Doc benchmark on Modal.

    modal run --detach eval_modal/run_benchmark.py --tag tier1

Runs ``scripts/run_mmlongbench_benchmark.py`` unchanged inside the container, reading the
store that ``eval_modal/ingest.py`` built on the same volume and writing reports to
``/vol/results``. By default only questions whose document is ingested are run
(``--only-ingested``; pass ``--all-docs`` to require every document). The runner's memory
preflight and slow-question guard stay on, and the script is checkpointed and resumable.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal  # noqa: E402

from eval_modal.common import run_script, split_args  # noqa: E402
from eval_modal.image import (  # noqa: E402
    APP_PREFIX, CPU, MEMORY_MIB, RESULTS, STORE, TIMEOUT_SECONDS, VOLUME_NAME,
    image, secret, volume, volumes,
)

app = modal.App(f"{APP_PREFIX}-run", image=image)


@app.function(cpu=CPU, memory=MEMORY_MIB, timeout=TIMEOUT_SECONDS, retries=0,
              volumes=volumes, secrets=[secret()])
def run_benchmark(tag: str = "", extractor: str = "deepseek", scope: str = "pooled",
                  ids: str = "", limit: int = 0, all_docs: bool = False,
                  allow_missing: bool = False, no_resume: bool = False,
                  dry_run: bool = False, extra_args: str = "") -> dict:
    volume.reload()  # see what the ingest job committed
    args = ["--store", STORE, "--out-dir", RESULTS, "--extractor", extractor, "--scope", scope]
    if tag:
        args += ["--tag", tag]
    if ids:
        args += ["--ids", ids]
    if limit:
        args += ["--limit", str(limit)]
    if not all_docs:
        args.append("--only-ingested")
    for flag, on in (("--allow-missing", allow_missing), ("--no-resume", no_resume), ("--dry-run", dry_run)):
        if on:
            args.append(flag)
    code = run_script("scripts/run_mmlongbench_benchmark.py", args + split_args(extra_args), volume)
    if code != 0:
        raise RuntimeError(
            f"the benchmark exited with code {code} "
            "(3 = the memory or slow-question guard stopped it at a checkpoint; 2 = refused to start). "
            "Progress is committed to the volume; re-run the same command to resume.")
    return dict(tag=tag, exit_code=code, results=RESULTS)


@app.local_entrypoint()
def main(tag: str = "", extractor: str = "deepseek", scope: str = "pooled", ids: str = "",
         limit: int = 0, all_docs: bool = False, allow_missing: bool = False,
         no_resume: bool = False, dry_run: bool = False, extra_args: str = ""):
    result = run_benchmark.remote(tag, extractor, scope, ids, limit, all_docs,
                                  allow_missing, no_resume, dry_run, extra_args)
    print(result)
    print(f"fetch the reports:  modal volume get {VOLUME_NAME} results/ ./eval_results/")
