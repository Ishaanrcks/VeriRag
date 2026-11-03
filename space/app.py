"""VeriRAG demo — a Hugging Face Space wrapper around the shipping pipeline.

Upload a PDF, ask a question, and see the VERIFIED answer plus whether the system
abstained or refused. The pipeline runs exactly as it ships (final-c2: visual
verification on, two-gate critic). This file is UI and demo-safety only; all
pipeline integration is in demo_helpers.py, and nothing here touches the verirag
package or its frozen config.

Run locally:  python app.py   (needs DEEPSEEK_API_KEY in the environment/.env)
"""

from __future__ import annotations

import datetime as _dt
import os
import threading

import gradio as gr

from demo_helpers import DemoError, run_pipeline, validate_pdf, validate_question

# ============================================================================
# DEMO CONFIG — all knobs in one place
# ============================================================================
FREE_QUERIES_PER_VISITOR = 5      # free queries on the server key before BYO-key
DAILY_BUDGET_QUERIES = 200        # total server-key queries/day, then everyone is BYO
MAX_PDF_MB = 10.0                 # upload size cap
MAX_PDF_PAGES = 30                # page-count cap (ingest cost scales with pages)
MAX_QUESTION_CHARS = 500          # question length cap
# ============================================================================

_NOTICE = (
    "### VeriRAG demo — self-verifying document QA\n"
    "Upload a PDF, ask a question. The system answers **only** what it can ground "
    "in the document, and abstains otherwise.\n\n"
    "⚠️ **Demo only — do not upload confidential documents.** Uploads are processed "
    "in a throwaway per-session index and deleted after each query; nothing you "
    "upload is stored or visible to anyone else."
)

# ---- global daily budget on the server key (the real spend protection) ------
# Simplest reliable approach: an in-process counter under a lock, reset at
# midnight (server local time). It resets if the Space restarts -- acceptable for
# a demo, since a restart cannot increase total spend, only forgive past usage.
# To make this hard (survive restarts, bound true daily spend), swap this block
# for a tiny persistent store: a row in a SQLite file on the Space's persistent
# disk, or a Redis INCR with a daily-expiring key. The rest of the app is unchanged.
_budget_lock = threading.Lock()
_budget = {"date": _dt.date.today(), "count": 0}


def _budget_allows() -> bool:
    with _budget_lock:
        today = _dt.date.today()
        if _budget["date"] != today:
            _budget["date"], _budget["count"] = today, 0
        return _budget["count"] < DAILY_BUDGET_QUERIES


def _budget_consume() -> None:
    with _budget_lock:
        _budget["count"] += 1


def _answer(pdf, question, free_used, byo_key):
    """Gradio handler. Returns (status_md, verified_md, candidate_md, new_free_used).

    free_used is the per-visitor counter persisted in gr.BrowserState (localStorage).
    """
    free_used = int(free_used or 0)
    byo_key = (byo_key or "").strip()
    try:
        if pdf is None:
            raise DemoError("Please upload a PDF first.")
        pages = validate_pdf(pdf, MAX_PDF_MB, MAX_PDF_PAGES)
        question = validate_question(question, MAX_QUESTION_CHARS)

        # Who pays? Free path uses the server key until the per-visitor count OR the
        # global daily budget is exhausted; after that the visitor must bring a key.
        visitor_has_free = free_used < FREE_QUERIES_PER_VISITOR
        budget_ok = _budget_allows()
        use_backend = visitor_has_free and budget_ok

        if not use_backend and not byo_key:
            if not budget_ok:
                msg = ("The demo's shared daily budget is used up for today. "
                       "Paste your own DeepSeek API key below to keep going — "
                       "it is used only for your request and never stored.")
            else:
                msg = (f"You've used your {FREE_QUERIES_PER_VISITOR} free queries. "
                       "Paste your own DeepSeek API key below to continue — "
                       "it is used only for your request and never stored.")
            return _status("key-needed", msg), "", "", free_used

        user_key = None if use_backend else byo_key
        result = run_pipeline(pdf, question, user_key=user_key)

        # Count usage only for a successful call on the SERVER key.
        if result.used_backend_key:
            _budget_consume()
            free_used += 1

        status_md = _status(result.status, _status_text(result.status, free_used))
        verified_md = f"### Verified answer\n\n{result.verified_answer}"
        candidate_md = ""
        if result.candidate_answer:
            reason = f" _(unverified because: {result.candidate_reason})_" if result.candidate_reason else ""
            candidate_md = (
                "### ⚠️ UNVERIFIED candidate answer\n"
                "_The pipeline could not ground this; shown only as a lead, not an answer._"
                f"{reason}\n\n{result.candidate_answer}"
            )
        return status_md, verified_md, candidate_md, free_used

    except DemoError as e:
        return _status("error", str(e)), "", "", free_used
    except Exception:  # noqa: BLE001 — never surface a stack trace to a visitor
        return _status("error", "Unexpected error. Please try again."), "", "", free_used


def _status(kind: str, text: str) -> str:
    icon = {"verified": "✅", "partial": "🟡", "abstained": "⚪", "refused": "🛑",
            "key-needed": "🔑", "error": "⚠️"}.get(kind, "•")
    return f"{icon} **{kind.replace('-', ' ').title()}** — {text}"


def _status_text(status: str, free_used: int) -> str:
    left = max(0, FREE_QUERIES_PER_VISITOR - free_used)
    tail = f" ({left} free queries left)" if left else " (free queries used up)"
    base = {
        "verified": "every claim is grounded in the document.",
        "partial": "some claims are grounded; see the verified-facts list in the answer.",
        "abstained": "the system declined — the document did not support an answer.",
        "refused": "the system declined to answer from the document.",
    }.get(status, "done.")
    return base + tail


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="VeriRAG demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(_NOTICE)
        # Per-visitor free-query counter, persisted in the browser's localStorage.
        # HONESTY: this is trivially bypassable — clearing site data or opening an
        # incognito window resets it to 0. It is a courtesy throttle, NOT a security
        # control. The real spend protection is the server-side daily budget above.
        free_used = gr.BrowserState(0, storage_key="verirag_free_used")

        with gr.Row():
            with gr.Column(scale=1):
                pdf = gr.File(label="PDF (demo only — no confidential docs)",
                              file_types=[".pdf"], type="filepath")
                question = gr.Textbox(label="Your question", lines=2,
                                      max_length=MAX_QUESTION_CHARS,
                                      placeholder="What does the document say about…?")
                byo_key = gr.Textbox(
                    label="Your DeepSeek API key (only needed after the free queries)",
                    type="password", placeholder="sk-…  — used only for your request, never stored")
                ask = gr.Button("Ask", variant="primary")
                gr.Markdown(
                    f"_Limits: ≤{MAX_PDF_MB:.0f} MB, ≤{MAX_PDF_PAGES} pages, "
                    f"≤{MAX_QUESTION_CHARS} chars, {FREE_QUERIES_PER_VISITOR} free queries._")
            with gr.Column(scale=1):
                status = gr.Markdown()
                verified = gr.Markdown()
                candidate = gr.Markdown()

        # concurrency_limit=1: requests are serialized (see demo_helpers), which is
        # what makes the per-request store/key swaps safe.
        ask.click(_answer, inputs=[pdf, question, free_used, byo_key],
                  outputs=[status, verified, candidate, free_used],
                  concurrency_limit=1)
    return demo


if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("WARNING: DEEPSEEK_API_KEY is not set — the free path will fail until it is.")
    build_demo().queue(max_size=16).launch()
