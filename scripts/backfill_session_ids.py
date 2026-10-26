"""Backfill session_id on chunks ingested before multi-tenant scoping existed.

The UI scopes retrieval/listing to shared content plus the current session,
where "shared" means ``session_id == SHARED_SESSION_ID``. Chunks ingested before
that field existed carry no session_id and would be invisible to a UI session.
This stamps every such chunk as shared (and gives it an ingested_at so the
cleanup sweep treats it as old-but-shared, i.e. never deletes it).

Idempotent and lightweight (a Chroma metadata update -- no models, no LLM).

    python scripts/backfill_session_ids.py            # default ./data/chroma
    python scripts/backfill_session_ids.py eval_store/queensland/chroma
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from verirag.config import SHARED_SESSION_ID, settings  # noqa: E402


def main() -> int:
    persist_dir = sys.argv[1] if len(sys.argv) > 1 else settings.chroma_persist_dir
    settings.chroma_persist_dir = persist_dir

    from verirag.retrieval.vector_store import get_collection

    collection = get_collection()
    result = collection.get(include=["metadatas"])
    ids = result.get("ids") or []
    metas = result.get("metadatas") or []

    to_update_ids: list[str] = []
    to_update_meta: list[dict] = []
    ts = time.time()
    for cid, meta in zip(ids, metas):
        meta = dict(meta or {})
        if meta.get("session_id"):
            continue
        meta["session_id"] = SHARED_SESSION_ID
        meta.setdefault("ingested_at", ts)
        to_update_ids.append(cid)
        to_update_meta.append(meta)

    if not to_update_ids:
        print(f"{persist_dir}: all {len(ids)} chunk(s) already tagged; nothing to do.")
        return 0

    collection.update(ids=to_update_ids, metadatas=to_update_meta)
    print(f"{persist_dir}: tagged {len(to_update_ids)} of {len(ids)} chunk(s) "
          f"as session_id={SHARED_SESSION_ID!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
