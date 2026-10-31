"""Ingest a tier of MMLongBench-Doc documents on Modal.

    modal run --detach eval_modal/ingest.py --tier tier1 --n 10

Runs ``scripts/ingest_mmlongbench_tier.py`` unchanged inside the container, against the
persistent volume: PDFs, the Chroma/BM25 store, ``ingest_log.json`` and the tier manifest
all live under ``/vol/mmlongbench``, so nothing is re-ingested between runs. The script's
own memory preflight stays on. Add ``--force`` to re-ingest documents already on the volume
(a clean replace with the current code); without it they are skipped. ``--detach`` lets the job keep running if this terminal
closes; ``retries`` is 0 because the job is checkpointed and stateful.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal  # noqa: E402

from eval_modal.common import run_script, split_args  # noqa: E402
from eval_modal.image import (  # noqa: E402
    APP_PREFIX, CPU, MANIFEST, MEMORY_MIB, STORE, TIMEOUT_SECONDS, VOLUME_NAME,
    image, secret, volume, volumes,
)

app = modal.App(f"{APP_PREFIX}-ingest", image=image)


@app.function(cpu=CPU, memory=MEMORY_MIB, timeout=TIMEOUT_SECONDS, retries=0,
              volumes=volumes, secrets=[secret()])
def ingest(tier: str, n: int, extends: str = "", max_mb: float = 4.0,
           max_pages: int = 100, force: bool = False, extra_args: str = "") -> dict:
    volume.reload()
    args = ["--tier", tier, "--n", str(n), "--store", STORE, "--manifest", MANIFEST,
            "--max-mb", str(max_mb), "--max-pages", str(max_pages)]
    if extends:
        args += ["--extends", extends]
    if force:
        args.append("--force")
    code = run_script("scripts/ingest_mmlongbench_tier.py", args + split_args(extra_args), volume)
    if code != 0:
        raise RuntimeError(
            f"ingestion exited with code {code} "
            "(3 = a memory guard stopped it; 1 = at least one document failed; 2 = bad arguments). "
            "Progress up to that point is committed to the volume; see ingest_log.json.")
    return dict(tier=tier, exit_code=code, store=STORE)


@app.local_entrypoint()
def main(tier: str, n: int = 10, extends: str = "", max_mb: float = 4.0,
         max_pages: int = 100, force: bool = False, extra_args: str = ""):
    result = ingest.remote(tier, n, extends, max_mb, max_pages, force, extra_args)
    print(result)
    print(f"fetch the log:  modal volume get {VOLUME_NAME} mmlongbench/ingest_log.json .")
