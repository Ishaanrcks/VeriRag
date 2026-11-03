"""VeriRAG backend on Modal — the heavy RAG pipeline behind an HTTP API.

The Gradio Space (space/app.py) is now a thin CPU frontend that calls this. All
the weight — embeddings, cross-encoder rerank, DeBERTa NLI, DeepSeek calls,
PyMuPDF ingest — runs here, on the image and Volume already defined in
eval_modal/image.py. Nothing in the verirag package or the frozen config changes.

State model (Option 2, stateful): one Modal Volume holds a single Chroma + BM25
store. The demo corpus is ingested once under the SHARED session id; each visitor's
uploads are ingested into the SAME store tagged with their session id; retrieval is
session-scoped, so a visitor sees the demo corpus plus only their own uploads. The
ASGI app runs as a SINGLE writer (max_containers=1) so concurrent requests cannot
corrupt the store, and ingests commit to the Volume for durability.

Endpoints (JSON unless noted):
  GET  /health            liveness + whether the demo corpus is loaded
  POST /ingest  (multipart file + session_id)   -> chunk counts
  POST /query   (json)    -> NDJSON stream: one {"step": ...} per pipeline stage,
                             then a final {"result": {...}}
  POST /cleanup (json session_id)               -> drop that session's chunks
  POST /corpus  (json session_id)               -> shared + session documents

Deploy:  modal deploy modal_backend.py
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import modal

# Reuse the existing heavy image, Volume and secret rather than redefining them.
from eval_modal.image import (
    REMOTE_REPO, REPO, VOLUME_PATH, base_image, secret, volume, volumes,
)
from eval_modal.mount_rules import IGNORE_PATTERNS

APP_NAME = "veri-rag-backend"
STORE = f"{VOLUME_PATH}/demo_store"          # dedicated demo store on the Volume
DEMO_CORPUS = "/root/veri-rag/demo_corpus"   # baked into the image (add_local_dir)
_PRELOAD_MARKER = f"{STORE}/.demo_preloaded"

# The backend needs FastAPI. It MUST be installed before the local source is
# mounted (Modal forbids build steps after add_local_*), so build from
# base_image (pre-local) and append our own add_local_dir last.
backend_image = (
    base_image.pip_install("fastapi[standard]==0.115.6")
    .add_local_dir(REPO, REMOTE_REPO, ignore=IGNORE_PATTERNS)
)

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# serialization: VeriRAGResult / citations / media / pipeline log -> plain JSON
# ---------------------------------------------------------------------------
def _serialize_media(citations) -> dict:
    """{chunk_id: {content_type, text, page, image_b64}} for table/figure chunks.

    Resolved HERE (the frontend has no store): tables as GFM markdown, figures as a
    base64 data URI of the saved crop so the Space can render them inline without a
    file server or Volume access."""
    from verirag.retrieval.vector_store import get_collection

    ids = []
    for c in citations:
        for cid in getattr(c, "contributing_chunk_ids", []) or []:
            if cid not in ids:
                ids.append(cid)
    if not ids:
        return {}
    try:
        got = get_collection().get(ids=ids, include=["documents", "metadatas"])
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for cid, text, meta in zip(got.get("ids") or [], got.get("documents") or [],
                               got.get("metadatas") or []):
        ct = meta.get("content_type")
        if ct not in ("table", "figure_caption"):
            continue
        entry = {"content_type": ct, "text": text, "page": meta.get("page"), "image_b64": None}
        if ct == "figure_caption":
            ip = meta.get("image_path")
            if ip and os.path.exists(str(ip)):
                try:
                    raw = Path(str(ip)).read_bytes()
                    if len(raw) <= 4_000_000:   # keep the payload sane
                        entry["image_b64"] = "data:image/png;base64," + base64.b64encode(raw).decode()
                except OSError:
                    pass
        out[cid] = entry
    return out


def _serialize_result(r) -> dict:
    """A VeriRAGResult as plain JSON the frontend renders without verirag."""
    def verdict(v):
        return None if v is None else {
            "verdict": getattr(v, "verdict", None),
            "overall_grounded": getattr(v, "overall_grounded", None),
        }
    citations = getattr(r, "citations", []) or []
    retrieved = getattr(r, "retrieved_citations", []) or []
    return {
        "answer": getattr(r, "answer", None),
        "abstained": bool(getattr(r, "abstained", False)),
        "outcome": getattr(r, "outcome", "verified"),
        "abstention_reason": getattr(r, "abstention_reason", None),
        "regeneration_count": getattr(r, "regeneration_count", 0),
        "llm_call_count": getattr(r, "llm_call_count", 0),
        "total_latency_ms": getattr(r, "total_latency_ms", 0.0),
        "estimated_cost_usd": getattr(r, "estimated_cost_usd", 0.0),
        "gate1_verdict": verdict(getattr(r, "gate1_verdict", None)),
        "gate2_verdict": verdict(getattr(r, "gate2_verdict", None)),
        "candidate_answer": getattr(r, "candidate_answer", None),
        "candidate_reason": getattr(r, "candidate_reason", None),
        "trace": list(getattr(r, "trace", []) or []),
        "citations": [_serialize_citation(c) for c in citations],
        "retrieved_citations": [_serialize_citation(c) for c in retrieved],
        "media": _serialize_media(list(citations) + list(retrieved)),
        # Pre-rendered on the backend: the pipeline-log markdown uses core helpers
        # (retrieval_rows/gate2_rows/...), which the frontend must not import.
        "pipelog_md": _render_pipelog_md(getattr(r, "pipeline_log", None) or []),
    }


def _render_pipelog_md(events) -> str:
    """The Pipeline Log tab's markdown, built here so the frontend stays verirag-free."""
    if not events:
        return "Run a query to see the pipeline trace."
    from verirag.pipeline_log import (abstain_parts, gate1_line, gate2_rows,
                                      included_lines, retrieval_rows)

    def of(stage):
        return [e for e in events if e.stage == stage]

    def table(rows, cols):
        if not rows:
            return "_(none)_"
        present = [c for c in cols if any(c in r for r in rows)]
        head = "| " + " | ".join(present) + " |"
        sep = "| " + " | ".join("---" for _ in present) + " |"
        body = ["| " + " | ".join(
            (f"{r.get(c):.3f}" if isinstance(r.get(c), float) else str(r.get(c, "")))
            .replace("\n", " ").replace("|", "\\|")[:200] for c in present) + " |"
            for r in rows]
        return "\n".join([head, sep, *body])

    out: list[str] = []
    rets = of("retrieval")
    if rets:
        ev = rets[-1]
        out.append(f"### Retrieval — {ev.detail['chunk_count']} chunk(s) from "
                   f"{ev.detail['doc_count']} document(s)")
        out.append(table(retrieval_rows(ev.detail["chunks"]),
                         ["Rank", "Page", "Type", "Score", "Document", "Snippet"]))
        out.append("_Score is the cross-encoder rerank score (the value Gate 1 reads)._")
    ctxs = of("context")
    if ctxs:
        ev = ctxs[-1]
        lines = included_lines(ev.detail["chunks_included"])
        out.append("### Final context sent to LLM")
        out.append(f"_{len(ev.detail['chunks_included'])} chunk(s) included:_")
        out += [f"- {ln}" for ln in lines[:15]]
        if len(lines) > 15:
            out.append(f"_... and {len(lines) - 15} more_")
    out.append("### Gate 1 — Coverage")
    g1 = of("gate1")
    if g1:
        for ev in g1:
            ok = ev.detail["verdict"] == "pass"
            out.append(("[OK] " if ok else "[!] ") + gate1_line(ev.detail))
    else:
        out.append("_Gate 1 did not run on this path._")
    regens = of("regen")
    if regens:
        out.append("### Regeneration")
        for ev in regens:
            d = ev.detail
            out.append(f"[i] Attempt {d['iteration']} of {d['max_iterations']} — "
                       f"triggered by: {d['reason']}")
    g2 = of("gate2")
    if g2:
        out.append("### Gate 2 — Groundedness")
        claims = g2[-1].detail["claims"]
        if claims:
            out.append(table(gate2_rows(claims), ["Claim", "Verdict", "Source Type", "Score"]))
            out.append("_Score: NLI prob. or token-overlap, by verdict type._")
        else:
            out.append("_No checkable claims were extracted from the answer._")
    out.append("### Result")
    ab, an = of("abstain"), of("answer")
    if ab:
        head, bullets = abstain_parts(ab[-1].detail["reason"])
        out.append(f"[!] **Abstained — {head}**")
        out += [f"- {b}" for b in bullets]
    elif an:
        out.append("[OK] **Answered**")
        out.append(an[-1].detail["text"])
    return "\n\n".join(out)


def _serialize_citation(c) -> dict:
    return {
        "number": getattr(c, "number", None),
        "source_path": getattr(c, "source_path", ""),
        "pages": list(getattr(c, "pages", []) or []),
        "content_types": list(getattr(c, "content_types", []) or []),
        "contributing_chunk_ids": list(getattr(c, "contributing_chunk_ids", []) or []),
        "url": getattr(c, "url", None),
        "source_type": getattr(c, "source_type", "local"),
        "parent_text": getattr(c, "parent_text", "") or "",
    }


# ---------------------------------------------------------------------------
# store setup + demo-corpus preload (idempotent)
# ---------------------------------------------------------------------------
def _configure() -> None:
    from verirag.eval.harness import configure_store
    configure_store(STORE)


def _ensure_demo_corpus() -> list[str]:
    """Ingest demo_corpus/ under SHARED once; a marker file on the Volume makes it
    a no-op afterwards."""
    from verirag.config import SHARED_SESSION_ID
    from verirag.ingest import ingest_file

    _configure()
    names: list[str] = []
    corpus = Path(DEMO_CORPUS)
    if corpus.is_dir():
        for entry in sorted(corpus.iterdir()):
            if entry.suffix.lower() not in (".pdf", ".txt", ".md", ".docx", ".pptx"):
                continue
            if entry.name.lower() in ("readme.md", "licenses.md"):
                continue
            try:
                ingest_file(entry, session_id=SHARED_SESSION_ID)
                names.append(entry.name)
            except Exception:  # noqa: BLE001
                pass
    Path(_PRELOAD_MARKER).parent.mkdir(parents=True, exist_ok=True)
    Path(_PRELOAD_MARKER).write_text("1")
    volume.commit()
    return names


# ---------------------------------------------------------------------------
# the ASGI app — single writer so the shared store stays consistent
# ---------------------------------------------------------------------------
@app.function(
    image=backend_image, volumes=volumes, secrets=[secret()],
    max_containers=1,              # single writer: no concurrent Chroma corruption
    scaledown_window=600,          # stay warm 10 min so the store stays hot
    timeout=1800,
)
@modal.concurrent(max_inputs=8)    # serve several HTTP reads from the one container
@modal.asgi_app()
def fastapi_app():
    import uuid as _uuid

    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import JSONResponse, StreamingResponse

    api = FastAPI(title="VeriRAG backend")

    volume.reload()
    _configure()
    if not Path(_PRELOAD_MARKER).exists():
        _ensure_demo_corpus()

    @api.get("/health")
    def health():
        return {"ok": True, "demo_loaded": Path(_PRELOAD_MARKER).exists(), "store": STORE}

    @api.post("/ingest")
    async def ingest(session_id: str = Form(...), file: UploadFile = File(...)):
        from verirag.ingest import ingest_file
        _configure()
        tmp = Path("/tmp") / f"{_uuid.uuid4().hex}_{os.path.basename(file.filename or 'upload')}"
        tmp.write_bytes(await file.read())
        try:
            res = ingest_file(tmp, session_id=session_id)
            volume.commit()
            return {"doc_id": res.doc_id, "num_text_chunks": res.num_text_chunks,
                    "num_figure_chunks": res.num_figure_chunks, "skipped": res.skipped}
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=400, content={"error": str(exc)[:300]})
        finally:
            tmp.unlink(missing_ok=True)

    @api.post("/corpus")
    async def corpus(payload: dict):
        from verirag.config import SHARED_SESSION_ID
        from verirag.retrieval.vector_store import list_ingested_documents
        _configure()
        sid = payload.get("session_id")
        rows = list_ingested_documents() or []
        keep = [r for r in rows if r.get("session_id") in (None, SHARED_SESSION_ID, sid)]
        return {"documents": keep}

    @api.post("/stats")
    async def stats(payload: dict | None = None):
        try:
            from verirag.observability.metrics import summarize_metrics
            s = summarize_metrics()
        except Exception as exc:  # noqa: BLE001
            return {"message": f"Metrics unavailable: {exc}"}
        if not s.total_queries:
            return {"message": "No queries logged yet. Ask something and these fill in."}
        return {"total": s.total_queries, "abstention_rate": s.abstention_rate,
                "loop_trigger_rate": s.loop_trigger_rate,
                "avg_cost": s.avg_cost_per_query_usd,
                "p95": sorted(s.p95_latency_by_node.items(), key=lambda kv: kv[1],
                              reverse=True)[:6]}

    @api.post("/cleanup")
    async def cleanup(payload: dict):
        from verirag.config import SHARED_SESSION_ID
        from verirag.retrieval import vector_store
        sid = payload.get("session_id")
        if sid and sid != SHARED_SESSION_ID:
            _configure()
            vector_store.delete_by_session(sid)
            volume.commit()
        return {"ok": True}

    @api.post("/query")
    async def query(payload: dict):
        from verirag.generation.generator import AnswerFormat
        from verirag.graph.build import run_query_streaming
        from verirag.llm.client import (QuotaExceededError, set_session_api_key,
                                        set_surface_quota)

        question = (payload.get("question") or "").strip()
        session_id = payload.get("session_id")
        fmt = getattr(AnswerFormat, (payload.get("format") or "CONCISE").upper(),
                      AnswerFormat.CONCISE)
        corpus_only = bool(payload.get("corpus_only", True))
        user_key = (payload.get("user_key") or "").strip() or None

        def stream():
            _configure()
            from verirag.config import settings
            prev = settings.deepseek_api_key
            set_surface_quota(True)
            if user_key:
                set_session_api_key(user_key)
                settings.deepseek_api_key = user_key
            try:
                gen = run_query_streaming(question, fmt, corpus_only=corpus_only,
                                          session_id=session_id)
                while True:
                    try:
                        _, summary = next(gen)
                        yield json.dumps({"step": summary}) + "\n"
                    except StopIteration as stop:
                        yield json.dumps({"result": _serialize_result(stop.value)}) + "\n"
                        return
            except QuotaExceededError:
                yield json.dumps({"quota": bool(user_key)}) + "\n"
            except Exception as exc:  # noqa: BLE001
                yield json.dumps({"error": str(exc)[:300]}) + "\n"
            finally:
                set_session_api_key(None)
                set_surface_quota(False)
                settings.deepseek_api_key = prev

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return api


@app.function(image=backend_image, volumes=volumes, secrets=[secret()], timeout=1800)
def preload_remote() -> list[str]:
    """Ingest the demo corpus into the Volume store (runs on Modal)."""
    return _ensure_demo_corpus()


@app.local_entrypoint()
def preload():
    """One-off: warm the demo corpus before the first request.
    `modal run modal_backend.py` (otherwise the first /health lazily preloads it)."""
    print("preloading demo corpus into the Volume store...")
    print("done:", preload_remote.remote())
