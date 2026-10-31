"""Modal health check: does the image build, does the volume mount, is the secret readable?

    modal run eval_modal/health.py

Costs a few cents: it builds the image once (cached afterwards) and runs four short
functions. Nothing is ingested and no API is called. The first function is Modal's own
"square of 42" example, so a pass means the image built and a container started; the
others check what the real jobs depend on. Secret VALUES are never printed, only
whether each key is present.

If the Modal Secret does not exist yet, the secret check reports PENDING (not FAIL)
and prints the command that creates it; everything else is still verified.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal  # noqa: E402

from eval_modal.image import (  # noqa: E402
    APP_PREFIX, CPU, MEMORY_MIB, MODEL_CACHE, MODELS, REMOTE_REPO, REQUIRED_SECRET_KEYS,
    SECRET_NAME, VOLUME_NAME, VOLUME_PATH, image, volume, volumes,
)

OPTIONAL_KEYS = ["GEMINI_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_CHAT_MODEL",
                 "DEEPSEEK_VISION_MODEL", "LANGSMITH_API_KEY", "TAVILY_API_KEY"]


def _existing_secret():
    """The project secret if it exists in this workspace, else ``(None, reason)``."""
    candidate = modal.Secret.from_name(SECRET_NAME, required_keys=REQUIRED_SECRET_KEYS)
    try:
        candidate.hydrate()
        return candidate, ""
    except Exception as exc:  # noqa: BLE001 - any failure to resolve means "not usable yet"
        return None, f"{type(exc).__name__}: {exc}"


_secret, _secret_problem = _existing_secret()
app = modal.App(f"{APP_PREFIX}-health", image=image)


@app.function()
def square(x: int) -> int:
    """Modal's own smoke test: if this returns, the image built and a container started."""
    return x * x


@app.function(volumes=volumes)
def volume_check() -> dict:
    """Mount the volume, write, commit, reload and read back; show what a past run left."""
    import os
    import time

    ping = Path(VOLUME_PATH) / "_health" / "ping.txt"
    previous = ping.read_text() if ping.exists() else None
    ping.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"health check at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
    ping.write_text(stamp)
    volume.commit()
    volume.reload()
    stat = os.statvfs(VOLUME_PATH)
    return dict(
        mounted=Path(VOLUME_PATH).is_dir(),
        wrote_and_read_back=ping.read_text() == stamp,
        previous_ping=previous,
        top_level=sorted(p.name for p in Path(VOLUME_PATH).iterdir()),
        free_gb=round(stat.f_bavail * stat.f_frsize / 1e9, 1),
    )


@app.function(cpu=CPU, memory=MEMORY_MIB, timeout=1200, retries=0, volumes=volumes,
              secrets=[_secret] if _secret else [])
def environment_check() -> dict:
    """Everything the real jobs depend on, with the same CPU and memory they will get."""
    import importlib
    import os
    import platform
    import time
    import urllib.request

    report: dict = {}
    report["python"] = platform.python_version()
    report["cpu_count"] = os.cpu_count()

    # secrets: presence only, never values
    present = {k: bool(os.environ.get(k)) for k in REQUIRED_SECRET_KEYS + OPTIONAL_KEYS}
    report["secret_keys_present"] = present
    report["secret_required_ok"] = all(present[k] for k in REQUIRED_SECRET_KEYS)
    # config.Settings needs DEEPSEEK_API_KEY just to import; a placeholder lets the
    # model checks run before the secret exists. It is never used to call an API.
    os.environ.setdefault("DEEPSEEK_API_KEY", "health-check-placeholder")

    # the repo mount: what must be there, and what must not
    root = Path(REMOTE_REPO)
    report["repo_has"] = {p: (root / p).exists() for p in (
        "src/verirag/config.py", "scripts/ingest_mmlongbench_tier.py",
        "scripts/run_mmlongbench_benchmark.py",
        "eval_data/mmlongbench_subset200_v1canonical.json", "requirements.txt")}
    report["env_file_absent"] = not (root / ".env").exists()
    report["git_dir_absent"] = not (root / ".git").exists()

    # dependency set
    import torch

    report["torch"] = dict(version=torch.__version__, cuda=torch.cuda.is_available(),
                           threads=torch.get_num_threads())
    imports = {}
    for module in ("verirag.config", "verirag.ingest", "verirag.graph.build",
                   "verirag.eval.mmlongbench_runner", "verirag.eval.mmlongbench_tiers",
                   "pymupdf", "chromadb", "sentence_transformers"):
        try:
            importlib.import_module(module)
            imports[module] = "ok"
        except Exception as exc:  # noqa: BLE001
            imports[module] = f"FAILED {type(exc).__name__}: {exc}"
    report["imports"] = imports

    # the memory gauge the runner's preflight uses, as it reads in THIS container
    from verirag.eval.mmlongbench_runner import (
        MIN_AVAILABLE_GB, check_memory, memory_summary, read_memory_status,
    )

    status = read_memory_status()
    report["memory_gauge"] = status.describe() if status else "unreadable (preflight fails open)"
    report["memory_preflight"] = memory_summary(MIN_AVAILABLE_GB)[0]

    # models are baked into the image: prove it by forbidding the network
    os.environ["HF_HUB_OFFLINE"] = "1"
    report["models_in_cache"] = sorted(p.name for p in Path(MODEL_CACHE, "hub").glob("models--*"))
    timings = {}
    try:
        from verirag.critic.nli import check_entailment_windowed
        from verirag.embeddings.embedder import embed_passages
        from verirag.retrieval.reranker import rerank
        from verirag.retrieval.types import SearchHit

        t = time.perf_counter()
        vectors = embed_passages(["The quick brown fox jumps over the lazy dog."])
        timings["embedder_s"] = round(time.perf_counter() - t, 1)
        report["embedding_dim"] = int(vectors.shape[1])
        hits = [SearchHit(chunk_id="a", raw_text="Paris is the capital of France.", metadata={}),
                SearchHit(chunk_id="b", raw_text="Bananas are yellow.", metadata={})]
        t = time.perf_counter()
        ranked = rerank("What is the capital of France?", hits, 2)
        timings["reranker_s"] = round(time.perf_counter() - t, 1)
        report["reranker_top"] = ranked[0].chunk_id
        t = time.perf_counter()
        entailment = check_entailment_windowed("A cat is sitting on the mat.", "A cat is on a mat.")
        timings["nli_s"] = round(time.perf_counter() - t, 1)
        report["nli_entailment_prob"] = round(float(entailment.entailment_prob), 3)
        report["models_offline_ok"] = (ranked[0].chunk_id == "a"
                                       and report["nli_entailment_prob"] > 0.5)
    except Exception as exc:  # noqa: BLE001
        report["models_offline_ok"] = False
        report["models_error"] = f"{type(exc).__name__}: {exc}"
    report["timings"] = timings
    status_after = read_memory_status()
    report["memory_gauge_with_models"] = status_after.describe() if status_after else "unreadable"

    # outbound network the jobs need (any HTTP answer, even an error status, = reachable)
    reach = {}
    for name, url in (("huggingface.co (PDF downloads)", "https://huggingface.co/api/datasets/yubo2333/MMLongBench-Doc"),
                      ("api.deepseek.com", "https://api.deepseek.com")):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=20) as r:
                reach[name] = r.status
        except urllib.error.HTTPError as exc:
            reach[name] = exc.code
        except Exception as exc:  # noqa: BLE001
            reach[name] = f"UNREACHABLE {type(exc).__name__}"
    report["network"] = reach
    return report


def _line(label: str, state: str, detail: str = "") -> str:
    return f"  {state:8} {label}" + (f"  -- {detail}" if detail else "")


@app.local_entrypoint()
def main():
    print(f"image: debian_slim + pinned requirements; volume {VOLUME_NAME!r}; secret {SECRET_NAME!r}\n")
    squared = square.remote(42)
    vol = volume_check.remote()
    env = environment_check.remote()

    bad_imports = {k: v for k, v in env["imports"].items() if v != "ok"}
    rows = [
        _line("image builds, container starts (square of 42)", "PASS" if squared == 1764 else "FAIL", f"42^2 = {squared}"),
        _line("volume mounts, write/commit/reload/read back", "PASS" if vol["mounted"] and vol["wrote_and_read_back"] else "FAIL",
              f"{vol['free_gb']} GB free; previous ping: {vol['previous_ping'] or 'none (first run)'}"),
        _line("dependencies import (project + torch + chromadb + pymupdf)", "PASS" if not bad_imports else "FAIL",
              f"torch {env['torch']['version']} cuda={env['torch']['cuda']} threads={env['torch']['threads']}" if not bad_imports else str(bad_imports)),
        _line("repo mounted; .env and .git NOT mounted", "PASS" if all(env["repo_has"].values()) and env["env_file_absent"] and env["git_dir_absent"] else "FAIL"),
        _line("3 models load offline from the image cache and run", "PASS" if env["models_offline_ok"] else "FAIL",
              f"{env['timings']} dim={env.get('embedding_dim')}" + (f" {env.get('models_error')}" if env.get("models_error") else "")),
        _line("memory gauge works in the container", "PASS" if "unreadable" not in env["memory_gauge"] else "WARN", env["memory_gauge"]),
        _line("network: HF + DeepSeek reachable", "PASS" if all(isinstance(v, int) for v in env["network"].values()) else "FAIL", str(env["network"])),
        _line(f"secret {SECRET_NAME!r} readable (required keys present)",
              "PASS" if _secret and env["secret_required_ok"] else "PENDING",
              "keys: " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in env["secret_keys_present"].items())
              if _secret else f"not found in this workspace ({_secret_problem[:90]})"),
    ]
    print("\n".join(rows))
    print(f"\ncontainer: {env['cpu_count']} cpus, memory {env['memory_gauge']}")
    print(f"preflight in container: {env['memory_preflight']}")
    print(f"memory with the 3 models resident: {env['memory_gauge_with_models']}")
    print(f"models cached in image: {env['models_in_cache']}")
    if not _secret:
        print(f"\nTo create the secret (it uploads your local .env to Modal, so every key in it is stored there):\n"
              f"  modal secret create {SECRET_NAME} --from-dotenv .env")
    if any(r.split()[0] == "FAIL" for r in rows):
        raise SystemExit(1)
