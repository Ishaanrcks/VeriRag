"""The Modal image, volume and secret shared by every MMLongBench job.

One image holds the whole project: the pinned Python dependencies (CPU torch,
transformers, sentence-transformers, chromadb, PDF parsing and the OpenAI-compatible
client used for DeepSeek), the three small models baked into the image so no
container ever downloads them, and the repository itself mounted at /root/veri-rag.
No GPU: the embedder, reranker and NLI model are CPU-sized, so the functions ask for
generous CPU and memory instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # so `modal run eval_modal/<file>.py` works from any directory

import modal  # noqa: E402

from eval_modal.mount_rules import IGNORE_PATTERNS, requirements_for_container  # noqa: E402

APP_PREFIX = "veri-rag-mmlongbench"
VOLUME_NAME = "veri-rag-mmlongbench"
VOLUME_PATH = "/vol"
SECRET_NAME = "veri-rag-secrets"
# The one key the project cannot start without (config.Settings.deepseek_api_key).
REQUIRED_SECRET_KEYS = ["DEEPSEEK_API_KEY"]

REMOTE_REPO = "/root/veri-rag"
MODEL_CACHE = "/models"
# The models named in src/verirag/config.py; baked into the image at build time.
MODELS = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-reranker-base",
    "cross-encoder/nli-deberta-v3-base",
]
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

# Layout on the volume
STORE = f"{VOLUME_PATH}/mmlongbench"          # chroma, bm25 index, PDFs, ingest_log.json
MANIFEST = f"{VOLUME_PATH}/mmlongbench/tiers.json"
RESULTS = f"{VOLUME_PATH}/results"

# Resources for the real jobs (the health check requests the same, to prove they fit).
CPU = 4
MEMORY_MIB = 12 * 1024
TIMEOUT_SECONDS = 12 * 60 * 60  # ingestion of a large tier is slow; retries stay 0 (checkpointed)

_download_models = (
    "from huggingface_hub import snapshot_download; "
    + "; ".join(f"snapshot_download('{m}')" for m in MODELS)
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        *requirements_for_container(REPO / "requirements.txt"),
        extra_index_url=CPU_TORCH_INDEX,
    )
    .env(
        {
            "HF_HOME": MODEL_CACHE,
            "PYTHONPATH": f"{REMOTE_REPO}/src",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .run_commands(f'python -c "{_download_models}"')
    .workdir(REMOTE_REPO)
    # last: local files are mounted when the container starts, so code edits need no rebuild
    .add_local_dir(REPO, REMOTE_REPO, ignore=IGNORE_PATTERNS)
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
volumes = {VOLUME_PATH: volume}


def secret() -> modal.Secret:
    """The project's secret; Modal verifies the required keys exist at start-up."""
    return modal.Secret.from_name(SECRET_NAME, required_keys=REQUIRED_SECRET_KEYS)
