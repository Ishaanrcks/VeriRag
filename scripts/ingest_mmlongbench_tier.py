"""Select, download and ingest one tier of MMLongBench-Doc documents, sequentially.

    python scripts/ingest_mmlongbench_tier.py --tier tier1 --n 10 --store eval_store/mmlongbench

Tiers nest: ``--extends tier1`` picks the next documents that add the most coverage
(see eval/mmlongbench_tiers.py). Documents are downloaded from the benchmark's
HuggingFace repo into ``<store>/docs`` and ingested ONE AT A TIME into ``<store>``;
both live under ``eval_store/``, which is gitignored (the PDFs must not be committed).

Memory is checked before the first document and before every later one, and the
script stops (exit 3) rather than thrash. A document that fails to download or to
ingest is logged and skipped, never silently: the log and the exit code (1) say so.
Everything is written to ``<store>/ingest_log.json`` after every document.

Run it alone: it loads the embedding model and (via captioning/contextualisation)
spends DeepSeek calls, and two model-loading processes wedge an 8 GB machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from verirag.eval.mmlongbench_runner import (
    MIN_AVAILABLE_GB,
    MIN_AVAILABLE_GB_DURING,
    MemoryLow,
    check_memory,
)
from verirag.eval.mmlongbench_tiers import DEFAULT_MAX_BYTES, coverage, rank_documents

HF = "https://huggingface.co/datasets/yubo2333/MMLongBench-Doc"


def hf_sizes() -> dict[str, int]:
    tree = json.load(urllib.request.urlopen(f"{HF.replace('/datasets/', '/api/datasets/')}/tree/main/documents"))
    return {item["path"].split("/")[-1]: item["size"] for item in tree}


def download(name: str, dest: Path, expected: int) -> None:
    """Fetch one PDF, verify it is a PDF of the advertised size."""
    if dest.exists() and dest.stat().st_size == expected:
        return
    url = f"{HF}/resolve/main/documents/{urllib.parse.quote(name)}"
    tmp = dest.with_suffix(".part")
    last: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            if tmp.stat().st_size != expected or tmp.read_bytes()[:5] != b"%PDF-":
                raise ValueError(f"downloaded {tmp.stat().st_size} bytes, expected {expected}, or not a PDF")
            tmp.rename(dest)
            return
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download of {name} failed: {last}")


def page_count(path: Path) -> int:
    import pymupdf

    with pymupdf.open(path) as pdf:
        return pdf.page_count


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", required=True, help="name of this tier, e.g. tier1")
    parser.add_argument("--n", type=int, required=True, help="documents to add")
    parser.add_argument("--extends", help="an earlier tier in the manifest to build on")
    parser.add_argument("--subset", default="eval_data/mmlongbench_subset200_v1canonical.json")
    parser.add_argument("--store", required=True)
    parser.add_argument("--manifest", default="eval_data/mmlongbench_tiers.json")
    parser.add_argument("--max-mb", type=float, default=DEFAULT_MAX_BYTES / 1e6)
    parser.add_argument("--max-pages", type=int, default=100,
                        help="skip a document longer than this (decided after download)")
    args = parser.parse_args()

    # 1. memory, before anything is loaded
    try:
        status = check_memory(MIN_AVAILABLE_GB, stage="start")
    except MemoryLow as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 3
    print(f"memory: {status.describe() if status else 'unreadable'} -- OK to start", flush=True)

    store = Path(args.store)
    docs_dir = store / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    log_path = store / "ingest_log.json"
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    questions = json.loads(Path(args.subset).read_text())["questions"]

    # 2. choose documents (download in rank order until n acceptable ones are in hand)
    earlier = manifest.get(args.extends, {}).get("documents", []) if args.extends else []
    if args.extends and not earlier:
        print(f"error: tier {args.extends!r} is not in {manifest_path}", file=sys.stderr)
        return 2
    sizes = hf_sizes()
    ranked = rank_documents(questions, sizes, already=earlier,
                            max_bytes=int(args.max_mb * 1e6))
    accepted: list[dict] = []
    rejected: list[dict] = []
    t_download = time.time()
    for name in ranked:
        if len(accepted) >= args.n:
            break
        try:
            download(name, docs_dir / name, sizes[name])
            pages = page_count(docs_dir / name)
        except Exception as exc:  # noqa: BLE001 - recorded, never silent
            rejected.append(dict(doc_id=name, reason=f"download/open failed: {exc}"))
            print(f"  skip {name}: {exc}", flush=True)
            continue
        if pages > args.max_pages:
            rejected.append(dict(doc_id=name, reason=f"{pages} pages > --max-pages {args.max_pages}"))
            print(f"  skip {name}: {pages} pages", flush=True)
            continue
        accepted.append(dict(doc_id=name, pages=pages, bytes=sizes[name]))
        print(f"  [{len(accepted)}/{args.n}] {name}  {pages} pages  {sizes[name] / 1e6:.1f} MB", flush=True)
    download_seconds = round(time.time() - t_download, 1)

    tier_docs = earlier + [d["doc_id"] for d in accepted]
    manifest[args.tier] = dict(
        documents=tier_docs, added=[d["doc_id"] for d in accepted], extends=args.extends,
        pages={d["doc_id"]: d["pages"] for d in accepted},
        rejected=rejected, coverage=coverage(questions, tier_docs),
        selection=("greedy coverage of question groups, cross-page and document type, then "
                   f"question count, then smaller size; files <= {args.max_mb:g} MB and "
                   f"<= {args.max_pages} pages; see eval/mmlongbench_tiers.py"),
        selected_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json(manifest_path, manifest)
    print(f"tier {args.tier}: {len(tier_docs)} documents, {manifest[args.tier]['coverage']['questions']} "
          f"questions reference them; downloads took {download_seconds}s", flush=True)

    # 3. ingest, strictly one at a time. images_dir must be set BEFORE the PDF loader
    # is imported (it reads it at import time) so figure images stay under the store.
    from verirag.config import settings
    from verirag.eval.harness import configure_store

    configure_store(store)
    settings.images_dir = str(store / "images")
    from verirag.ingest import ingest_file
    from verirag.retrieval import vector_store

    log = json.loads(log_path.read_text()) if log_path.exists() else {"documents": []}
    log.update(tier=args.tier, started_at=datetime.now(timezone.utc).isoformat())
    done = {d["doc_id"] for d in log["documents"] if d.get("status") == "ingested"}
    failures = 0
    t_all = time.time()
    for n, doc in enumerate(accepted, 1):
        name = doc["doc_id"]
        if name in done:
            print(f"[{n}/{len(accepted)}] {name}: already ingested, skipping", flush=True)
            continue
        try:
            check_memory(MIN_AVAILABLE_GB if n == 1 and not done else MIN_AVAILABLE_GB_DURING,
                         stage="start" if n == 1 and not done else "during")
        except MemoryLow as exc:
            write_json(log_path, log)
            print(f"stopped before {name}: {exc}", file=sys.stderr)
            return 3
        started = time.perf_counter()
        entry = dict(doc_id=name, pages=doc["pages"], bytes=doc["bytes"])
        try:
            result = ingest_file(docs_dir / name)
            collection = vector_store.get_collection().get(
                where={"doc_id": result.doc_id}, include=["metadatas"])
            metas = collection["metadatas"] or []
            entry.update(
                status="ingested", store_doc_id=result.doc_id, skipped=result.skipped,
                chunks=Counter(m.get("content_type") for m in metas),
                table_strategy=dict(Counter(m.get("table_strategy") for m in metas
                                            if m.get("content_type") == "table")),
                caption_truncated=sum(1 for m in metas if m.get("caption_truncated")),
            )
        except Exception as exc:  # noqa: BLE001 - recorded and reported, never routed around
            failures += 1
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        entry["seconds"] = round(time.perf_counter() - started, 1)
        log["documents"] = [d for d in log["documents"] if d["doc_id"] != name] + [entry]
        write_json(log_path, log)
        print(f"[{n}/{len(accepted)}] {name}: {entry['status']} in {entry['seconds']}s "
              f"{dict(entry.get('chunks', {}))}", flush=True)

    log["ingest_seconds"] = round(time.time() - t_all, 1)
    log["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(log_path, log)
    print(f"ingestion finished: {len(accepted) - failures} ok, {failures} failed, "
          f"{log['ingest_seconds']}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
