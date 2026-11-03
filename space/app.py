"""VeriRAG — Gradio demo for Hugging Face Spaces.

A deployment wrapper around the shipping pipeline (final-c2: visual verification
on, two-gate critic). Built to match the Streamlit UI (src/verirag/ui/app.py)
feature-for-feature. All demo logic is here and in demo_helpers.py; the verirag
package and its frozen config are untouched.

Run locally:  python app.py   (needs DEEPSEEK_API_KEY in the environment/.env)
"""

from __future__ import annotations

import datetime as _dt
import html
import os
import threading

import gradio as gr

# --- workaround for a gradio_client bug (present through 5.15.x) --------------
# Building the API schema at startup crashes with "argument of type 'bool' is not
# iterable" when any component's JSON schema contains a boolean node (e.g.
# additionalProperties: true/false) -- which gr.BrowserState triggers. This makes
# the whole app return HTTP 500 on first load. Guarding the recursive converter to
# treat a bool schema as "Any" fixes it without changing any behaviour. Demo-only;
# patches the gradio_client helper, never the verirag package.
import gradio_client.utils as _gcu  # noqa: E402

_orig_json_schema = _gcu._json_schema_to_python_type


def _json_schema_safe(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema(schema, defs)


_gcu._json_schema_to_python_type = _json_schema_safe
# -----------------------------------------------------------------------------

import demo_helpers as H
from demo_helpers import DemoError

# ============================================================================
# DEMO CONFIG — all knobs here
# ============================================================================
FREE_QUERIES = 5                  # free server-key queries per visitor (BrowserState)
DAILY_BUDGET_QUERIES = 200        # total server-key queries/day, then everyone is BYO
MAX_PDF_MB = 10.0
MAX_PDF_PAGES = 30
MAX_QUESTION_CHARS = 500
SHOW_CANDIDATE = os.getenv("UNVERIFIED_CANDIDATE", "false").lower() == "true"
# ============================================================================

FORMATS = {"Concise": "CONCISE", "Detailed": "DETAILED"}

DEMO_DOCS = [
    ("Alice's Adventures in Wonderland — literary text", [
        "What does Alice find on the little glass table after falling down the rabbit-hole?",
        "Who does Alice meet at the Mad Tea-Party?",
        "What is the price of the White Rabbit's pocket watch?"]),
    ("CDC NCHS Data Brief No. 529 — COPD report (tables & figures)", [
        "What percentage of adults aged 18 and over had COPD, by age group?",
        "How did COPD prevalence differ by sex?",
        "What is the recommended treatment for COPD according to this brief?"]),
    ("Semantic Versioning 2.0.0 — technical spec", [
        "When should the MAJOR version number be incremented?",
        "What do the three parts of a MAJOR.MINOR.PATCH version number mean?",
        "Which programming languages does the spec recommend?"]),
]

NOTICE = (
    "# VeriRAG\n"
    "A self-verifying document QA system with a two-gate verification critic. "
    "It answers from your documents with citations, or abstains.\n\n"
    "[!] **Demo only — do not upload confidential documents.** Uploads are private "
    "to your session and cleared automatically."
)

_DEMO_NAMES: list[str] = []

# ---- global daily budget on the server key (the real spend protection) ------
# In-memory counter under a lock, reset at midnight (server time). Resets on a
# Space restart too — acceptable for a demo (a restart cannot increase spend).
# To make it hard, swap this for a SQLite row on the Space's persistent disk or a
# Redis INCR with a daily-expiring key; nothing else changes.
_budget_lock = threading.Lock()
_budget = {"date": _dt.date.today(), "count": 0}


def _budget_allows() -> bool:
    with _budget_lock:
        if _budget["date"] != _dt.date.today():
            _budget["date"], _budget["count"] = _dt.date.today(), 0
        return _budget["count"] < DAILY_BUDGET_QUERIES


def _budget_consume() -> None:
    with _budget_lock:
        _budget["count"] += 1


# ---------------------------------------------------------------------------
# sidebar handlers
# ---------------------------------------------------------------------------
def _corpus_md(session_id: str) -> str:
    rows = H.list_corpus(session_id)
    if not rows:
        return "**Your corpus**\n\nNothing ingested yet."
    lines = [f"**Your corpus** — {len(rows)} document(s)"]
    for r in rows:
        from pathlib import Path
        icon = "(external)" if r.get("source_type") == "external" else "(local)"
        name = Path(str(r.get("source_path", ""))).name or r.get("source_path", "?")
        scope = "demo" if r.get("session_id") == H.SHARED_SESSION_ID else "yours"
        lines.append(f"- {icon} **{name}**  \n  `{str(r.get('doc_id',''))[:20]}` · "
                     f"{r.get('format','?')} · _{scope}_")
    return "\n".join(lines)


def _ingest(files, session_id):
    if not files:
        return "Nothing to ingest.", _corpus_md(session_id)
    msgs = []
    for f in files:
        path = f if isinstance(f, str) else getattr(f, "name", None)
        name = os.path.basename(path or "file")
        try:
            H.validate_upload(path, MAX_PDF_MB, MAX_PDF_PAGES)
            res = H.ingest_upload(path, session_id)
        except DemoError as e:
            msgs.append(f"[!] {e}")
            continue
        except Exception as e:  # noqa: BLE001
            msgs.append(f"[!] {name}: {H._friendly(e)}")
            continue
        if getattr(res, "skipped", False):
            msgs.append(f"[i] {name}: already in corpus (skipped)")
        else:
            msgs.append(f"[OK] {name}: {res.num_text_chunks} text + "
                        f"{res.num_figure_chunks} figure chunk(s)")
    return "\n\n".join(msgs), _corpus_md(session_id)


def _stats_md() -> str:
    s = H.system_stats()
    if isinstance(s, str):
        return f"**System stats**\n\n{s}"
    lines = [f"**System stats** — across {s['total']} logged quer"
             f"{'y' if s['total'] == 1 else 'ies'}",
             f"- Abstention rate: **{s['abstention_rate']:.0%}**",
             f"- Self-correction triggered: **{s['loop_trigger_rate']:.0%}**",
             f"- Avg cost / query: **${s['avg_cost']:.4f}**"]
    if s["p95"]:
        lines.append("- p95 latency by stage (ms):")
        lines += [f"    - {node}: {val:.0f}" for node, val in s["p95"]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main query handler (streaming generator)
# ---------------------------------------------------------------------------
def _gate_badge(label: str, passed: bool) -> str:
    color = "#1b5e20" if passed else "#b71c1c"   # real pass/fail, not hardcoded green
    word = "PASS" if passed else "FAIL"
    return (f"<span style='background:{color};color:#fff;padding:3px 9px;"
            f"border-radius:11px;font-size:0.82em'>{label} {word}</span>")


import re as _re


def _clean_table_cell(text: str) -> str:
    """Strip PDF text-extraction artefacts from table content: leader dots, merged
    CamelCase words, and runs of whitespace."""
    if not isinstance(text, str):
        return str(text)
    text = _re.sub(r"(\.\s*){3,}", " ", text)
    text = _re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = _re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _provenance(content_types) -> str:
    cts = content_types or []
    labels = [n for n, key in (("text", "text"), ("figure", "figure_caption"),
                               ("table", "table")) if key in cts]
    return " + ".join(labels) or "text"


def _render_result(query, r: dict) -> tuple[str, str]:
    """(answer_markdown, sources_markdown) from the backend's JSON result dict."""
    outcome = r.get("outcome", "verified")

    if r.get("abstained"):
        if outcome == "abstained_ungrounded":
            head = ("[blocked] **Could not verify the answer.** VeriRAG found relevant material "
                    "and drafted an answer, but at least one claim could not be verified — "
                    "withheld rather than shown unverified.")
        elif outcome == "abstained_insufficient":
            head = ("[--] **Not enough relevant information.** VeriRAG declined rather than "
                    "answering from general knowledge.")
        else:
            head = "[--] **VeriRAG declined to answer.**"
        why = r.get("abstention_reason") or "No reason recorded."
        ans = f"{head}\n\n<details open><summary><b>Why</b></summary>\n\n{why}\n\n</details>"
        srcs = _excerpts_md(r.get("retrieved_citations") or []) if outcome == "abstained_ungrounded" else ""
        return ans, srcs

    parts = []
    if outcome == "partial":
        parts.append("[partial] **Partially verified.** Some statements could not be verified "
                     "against the sources; only those under *Verified facts* are confirmed — "
                     "treat the rest with caution.")
    parts.append("### Answer\n\n" + (r.get("answer") or ""))

    g1, g2 = r.get("gate1_verdict") or {}, r.get("gate2_verdict") or {}
    g1_ok = g1.get("verdict") == "sufficient"
    g2_ok = bool(g2.get("overall_grounded"))
    badges = (_gate_badge("Gate 1 coverage", g1_ok) +
              " <span style='color:#888'>-&gt;</span> " +
              _gate_badge("Gate 2 groundedness", g2_ok))
    corrections = r.get("regeneration_count", 0)
    verdict_word = "Partially verified" if outcome == "partial" else "Verified"
    parts.append(badges + f"\n\n_{verdict_word} · {r.get('total_latency_ms', 0) / 1000:.1f}s · "
                 f"{corrections} self-correction{'' if corrections == 1 else 's'} · "
                 f"~${r.get('estimated_cost_usd', 0):.4f}_")

    if SHOW_CANDIDATE and r.get("candidate_answer"):
        reason = r.get("candidate_reason")
        parts.append("### [!] Unverified candidate (not quality-checked)\n\n"
                     + (f"_{reason}_\n\n" if reason else "") + r["candidate_answer"])

    return "\n\n".join(parts), _sources_md(r.get("citations") or [], r.get("media") or {})


def _sources_md(citations: list[dict], media: dict) -> str:
    if not citations:
        return ""
    out = ["### Sources"]
    for c in citations:
        prov = _provenance(c.get("content_types"))
        if c.get("source_type") == "external":
            out.append(f"**[{c['number']}]** *external* — [{c.get('url')}]({c.get('url')}) · {prov}")
        else:
            pages = ", ".join(str(p) for p in (c.get("pages") or []))
            ptxt = f" · page(s) {pages}" if pages else ""
            out.append(f"**[{c['number']}]** *local* — `{c.get('source_path','')}`{ptxt} · {prov}")
        for cid in c.get("contributing_chunk_ids") or []:
            m = media.get(cid)
            if not m:
                continue
            if m["content_type"] == "table" and m.get("text"):
                out.append(f"_Table — page {m.get('page')}_\n\n{_clean_table_cell(m['text'])}")
            elif m["content_type"] == "figure_caption":
                if m.get("image_b64"):          # backend sent the crop as a data URI
                    out.append(f"![figure]({m['image_b64']})\n\n_Figure — page {m.get('page')}_")
                elif m.get("text"):
                    out.append(f"_Figure — page {m.get('page')}_: {m['text']}")
    return "\n\n".join(out)


def _excerpts_md(citations: list[dict]) -> str:
    if not citations:
        return ""
    out = ["### Relevant excerpts from your documents",
           "_The retrieved passages the generator saw. Shown because the answer "
           "could not be verified — inspect them yourself._"]
    for c in citations:
        out.append(f"> {html.escape((c.get('parent_text') or '')[:1200])}")
    return "\n\n".join(out)


def _trace_md(r: dict) -> str:
    trace = r.get("trace") or []
    head = (f"_{len(trace)} step(s) · {r.get('regeneration_count', 0)} regeneration(s) · "
            f"{r.get('llm_call_count', 0)} LLM call(s)_\n")
    lines = [head] + [f"{i}. {H.friendly_step(s)}  \n   _{s}_" for i, s in enumerate(trace, 1)]
    return "\n".join(lines)


def ask(question, pdf_format, corpus_only, free_used, byo_key, session_id):
    """Streaming handler. Yields (status, answer, sources, trace, pipelog_md, free_used).

    Each pipeline step updates the status AND the live Pipeline Log in real time;
    the final yield replaces the live log with the full formatted tables. Gradio
    shows a spinner on the button throughout. Breadcrumbs go to stderr. (The query
    can take ~30-120s on CPU for the first run — that slowness, not a streaming bug,
    is what looked like a hang earlier.)"""
    import sys
    def log(m):  # noqa: ANN001
        print(f"[ask] {m}", file=sys.stderr, flush=True)

    free_used = int(free_used or 0)
    byo_key = (byo_key or "").strip()
    try:
        question = H.validate_question(question, MAX_QUESTION_CHARS)
    except DemoError as e:
        yield (f"[!] {e}", "", "", "", "Run a query to see the pipeline trace.", free_used)
        return

    visitor_free = free_used < FREE_QUERIES
    use_backend = visitor_free and _budget_allows()
    if not use_backend and not byo_key:
        why = ("The demo's shared daily budget is used up for today."
               if not _budget_allows() else f"You've used your {FREE_QUERIES} free demo queries.")
        yield (f"[key] {why} Paste your own DeepSeek API key below to continue — used "
               "only for your session, never stored.", "", "", "",
               "Run a query to see the pipeline trace.", free_used)
        return
    user_key = None if use_backend else byo_key

    AnswerFormat = __import__("verirag.generation.generator",
                              fromlist=["AnswerFormat"]).AnswerFormat
    fmt = getattr(AnswerFormat, FORMATS.get(pdf_format, "CONCISE"))

    log(f"start: backend={use_backend} fmt={fmt} corpus_only={bool(corpus_only)}")
    live_log: list[str] = []          # live pipeline log
    result = None
    for kind, payload in H.stream_query(question, fmt, bool(corpus_only), session_id, user_key):
        if kind == "step":
            log(f"step: {payload}")
            live_log.append(f"**{payload}**")
            yield ("[running] Thinking...", "", "", "", "\n\n".join(live_log), free_used)
        elif kind == "__quota__":
            msg = ("[blocked] **API limit reached on your key.** Wait a moment and try again, or "
                   "use a different key." if payload else
                   "[blocked] **The demo is experiencing high traffic** and has hit its API quota. "
                   "Try again in a few minutes, or enter your own DeepSeek API key.")
            log("quota")
            yield (msg, "", "", "", "\n\n".join(live_log), free_used)
            return
        elif kind == "__error__":
            log(f"error: {payload}")
            yield (f"[!] Query failed: {payload}", "", "", "", "\n\n".join(live_log), free_used)
            return
        elif kind == "__result__":
            result = payload

    if result is None:
        log("no result")
        yield ("[!] Query failed: no result produced.", "", "", "",
               "\n\n".join(live_log), free_used)
        return

    if use_backend:
        _budget_consume()
        free_used += 1
    answer_md, sources_md = _render_result(question, result)
    status = "[OK] Done" + (f" · {FREE_QUERIES - free_used} free queries left" if use_backend else "")
    full_log = result.get("pipelog_md") or "Run a query to see the pipeline trace."
    log(f"done: outcome={result.get('outcome')}")
    yield (status, answer_md, sources_md, _trace_md(result), full_log, free_used)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');
*, body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
p, li, label { line-height: 1.6; letter-spacing: -0.01em; }
h1 { font-size: 1.5rem; font-weight: 700; }
h2 { font-size: 1.15rem; font-weight: 600; }
h3 { font-size: 1rem; font-weight: 600; }
code, pre { font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important; font-size: 0.85em; }
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="VeriRAG", theme=gr.themes.Soft(), fill_width=True, css=_CSS) as demo:
        session_id = gr.State(value=H.new_session_id())
        free_used = gr.BrowserState(0, storage_key="verirag_free_used")

        gr.Markdown(NOTICE)
        with gr.Row():
            # ---------------- sidebar ----------------
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Corpus")
                uploads = gr.File(
                    label="Upload documents (private to your session)",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".pptx", ".txt", ".md"], type="filepath")
                gr.Markdown("_PDF, DOCX, PPTX, TXT or MD. Figures in PDFs get captioned. "
                            "Only you can query what you upload; it is cleared automatically._")
                ingest_btn = gr.Button("Ingest uploaded files", interactive=False)
                ingest_msg = gr.Markdown()
                corpus_md = gr.Markdown()
                refresh_btn = gr.Button("Refresh corpus list", size="sm")
                with gr.Accordion("System stats", open=False):
                    stats_md = gr.Markdown()
                    stats_btn = gr.Button("Refresh stats", size="sm")

            # ---------------- main ----------------
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Ask"):
                        with gr.Accordion("New here? Sample documents & questions to try", open=False):
                            gr.Markdown("_Each set has a factual lookup, a broader data "
                                        "question, and one the document does NOT answer._")
                            demo_btns = []
                            for name, qs in DEMO_DOCS:
                                gr.Markdown(f"**{name}**")
                                for q in qs:
                                    b = gr.Button(q, size="sm")
                                    demo_btns.append((b, q))
                        question = gr.Textbox(label="Question", lines=2,
                                              max_length=MAX_QUESTION_CHARS,
                                              placeholder="What does the Q4 revenue chart show?")
                        with gr.Row():
                            fmt = gr.Dropdown(list(FORMATS), value="Concise", label="Answer format")
                            corpus_only = gr.Checkbox(
                                label="Corpus-only mode", value=False,
                                info="Use only your documents — no web search.")
                        free_md = gr.Markdown()
                        byo_key = gr.Textbox(label="Your DeepSeek API key (only after free queries)",
                                             type="password",
                                             placeholder="sk-...  — used only for your session, never stored")
                        with gr.Row():
                            ask_btn = gr.Button("Ask", variant="primary", interactive=False)
                            stop_btn = gr.Button("Stop query", variant="stop", visible=False)
                        status = gr.Markdown()
                        answer = gr.Markdown()
                        sources = gr.Markdown()
                        with gr.Accordion("Show reasoning trace", open=False):
                            trace = gr.Markdown()
                    with gr.Tab("Pipeline Log"):
                        pipelog_md = gr.Markdown("Run a query to see the pipeline trace.")

        # ---- wiring ----
        def _free_banner(used):
            used = int(used or 0)
            left = max(0, FREE_QUERIES - used)
            if left:
                return f"_Free demo queries remaining: {left} of {FREE_QUERIES}._"
            return ("**You've used your free demo queries.** Enter your own DeepSeek API "
                    "key above to continue — used only for your session, never stored.")

        demo.load(lambda sid: (_corpus_md(sid), _stats_md()),
                  inputs=session_id, outputs=[corpus_md, stats_md])
        demo.load(_free_banner, inputs=free_used, outputs=free_md)

        uploads.change(lambda f: gr.update(interactive=bool(f)), inputs=uploads, outputs=ingest_btn)
        question.change(lambda q: gr.update(interactive=bool((q or '').strip())),
                        inputs=question, outputs=ask_btn)

        ingest_btn.click(_ingest, inputs=[uploads, session_id], outputs=[ingest_msg, corpus_md])
        refresh_btn.click(_corpus_md, inputs=session_id, outputs=corpus_md)
        stats_btn.click(_stats_md, outputs=stats_md)

        for b, q in demo_btns:
            b.click(lambda q=q: q, outputs=question)

        # Stop button: show while a query runs, hide when it finishes or is cancelled.
        def _running():
            return gr.update(visible=True), gr.update(interactive=False)

        def _idle():
            return gr.update(visible=False), gr.update(interactive=True)

        _ask_io = dict(
            inputs=[question, fmt, corpus_only, free_used, byo_key, session_id],
            outputs=[status, answer, sources, trace, pipelog_md, free_used],
            concurrency_limit=1,
        )
        run = (ask_btn.click(_running, outputs=[stop_btn, ask_btn])
               .then(ask, **_ask_io))
        run.then(_idle, outputs=[stop_btn, ask_btn]).then(
            _free_banner, inputs=free_used, outputs=free_md)

        srun = (question.submit(_running, outputs=[stop_btn, ask_btn])
                .then(ask, **_ask_io))
        srun.then(_idle, outputs=[stop_btn, ask_btn]).then(
            _free_banner, inputs=free_used, outputs=free_md)

        # Stop cancels whichever generator is running, resets the status, re-idles.
        stop_btn.click(lambda: "[stopped] Query cancelled.", outputs=status,
                       cancels=[run, srun]).then(_idle, outputs=[stop_btn, ask_btn])

    return demo


if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("WARNING: DEEPSEEK_API_KEY is not set — the free path will fail until it is.")
    _DEMO_NAMES = H.startup()
    print(f"Demo corpus preloaded: {_DEMO_NAMES}")
    build_demo().queue(max_size=16).launch(server_name="0.0.0.0")
