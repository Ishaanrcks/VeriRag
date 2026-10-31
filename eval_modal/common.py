"""Run one of the project's existing scripts inside a Modal container.

The scripts (``scripts/ingest_mmlongbench_tier.py``, ``scripts/run_mmlongbench_benchmark.py``)
already do the work, checkpoint themselves, and carry the memory preflight and the
slow-question guard; this module only runs them as a subprocess and keeps the Modal
Volume committed.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

REMOTE_REPO = Path("/root/veri-rag")


def split_args(extra: str) -> list[str]:
    """``--flag value`` pass-through for anything the entrypoint does not name."""
    return shlex.split(extra or "")


def run_script(script: str, args: list[str], volume, *, commit_every: float = 120.0) -> int:
    """Run ``script`` with ``args``; commit the volume periodically and once at the end.

    The committed state is what survives the container, so the final commit is in a
    ``finally`` (it also runs after a timeout or an exception). The periodic commits
    mean a container that dies mid-run keeps the documents ingested so far; a commit
    taken while Chroma is mid-write could capture a torn database, which is why the
    scripts are resumable and the last full commit wins.
    """
    command = [sys.executable, "-u", str(REMOTE_REPO / script), *args]
    print("+", " ".join(shlex.quote(c) for c in command), flush=True)
    process = subprocess.Popen(command, cwd=REMOTE_REPO)
    try:
        while True:
            try:
                return process.wait(timeout=commit_every)
            except subprocess.TimeoutExpired:
                volume.commit()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
        volume.commit()
