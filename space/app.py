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
MAX_PDF_PAGES = 60                # arXiv papers with appendices routinely run 30-50
MAX_QUESTION_CHARS = 500
MAX_PAPER_SUGGESTIONS = 5        # fixed number of suggestion slots (Gradio needs it up front)
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


def ingest_url(url, session_id):
    """Fetch a document by URL and ingest it into this visitor's session."""
    if not (url or "").strip():
        return "[!] Paste a URL first."
    try:
        name, skipped = H.ingest_from_url(url, session_id, MAX_PDF_MB, MAX_PDF_PAGES)
    except DemoError as e:
        return f"[!] {e}"
    except Exception as e:  # noqa: BLE001
        return f"[!] {H._friendly(e)}"
    if skipped:
        return f"[i] **{name}**: already in corpus (skipped)"
    return f"[ok] Added **{name}** to your corpus."


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


def _missing_facts_md(mf) -> str:
    """The structured "Not verified" / "Missing from the documents" sections.
    Empty string when there is nothing to show (a fully verified answer)."""
    if mf is None:
        return ""
    out = []
    if mf.unverified_claims:
        out.append("### Not verified\n\nThe following statements could not be grounded "
                   "in the sources:\n\n" + "\n".join(f"- {c}" for c in mf.unverified_claims))
    if mf.missing_info:
        out.append("### Missing from the documents\n\n"
                   + "\n".join(f"- {m}" for m in mf.missing_info))
    return "\n\n".join(out)


def _strip_unverified_note(answer: str) -> str:
    """Drop the core's trailing prose note (partial.UNVERIFIED_NOTE); the
    structured sections replace it. The Verified facts list itself is untouched."""
    try:
        from verirag.generation.partial import UNVERIFIED_NOTE
    except Exception:  # noqa: BLE001
        return answer
    return answer.replace(UNVERIFIED_NOTE, "").rstrip()


def _render_result(query, r, mf=None) -> tuple[str, str]:
    """(answer_markdown, sources_markdown) for a VeriRAGResult. ``mf`` is its
    MissingFacts (computed/cached by demo_helpers when not passed)."""
    abstained = bool(getattr(r, "abstained", False))
    outcome = getattr(r, "outcome", "verified")
    if mf is None:
        mf = H.extract_missing_facts(r, question=query)
    missing_md = _missing_facts_md(mf)

    if abstained:
        if outcome == "abstained_ungrounded":
            head = ("[blocked] **Could not verify the answer.** VeriRAG found relevant material "
                    "and drafted an answer, but at least one claim could not be verified — "
                    "withheld rather than shown unverified.")
        elif outcome == "abstained_insufficient":
            head = ("[--] **Not enough relevant information.** VeriRAG declined rather than "
                    "answering from general knowledge.")
        else:
            head = "[--] **VeriRAG declined to answer.**"
        why = getattr(r, "abstention_reason", None) or "No reason recorded."
        if mf.unverified_claims:
            # The reason lists the same claims as "- ..." lines; the structured
            # "Not verified" section below shows them, so keep only the headline.
            from verirag.pipeline_log import abstain_parts
            why, _ = abstain_parts(why)
        ans = f"{head}\n\n<details open><summary><b>Why</b></summary>\n\n{why}\n\n</details>"
        if missing_md:
            ans += "\n\n" + missing_md
        srcs = ""
        if outcome == "abstained_ungrounded":
            srcs = _excerpts_md(getattr(r, "retrieved_citations", []) or [])
        return ans, srcs

    parts = []
    if outcome == "partial":
        parts.append("[partial] **Partially verified.** Some statements could not be verified "
                     "against the sources; only those under *Verified facts* are confirmed — "
                     "treat the rest with caution.")
    parts.append("### Answer\n\n" + _strip_unverified_note(getattr(r, "answer", None) or ""))
    if missing_md:
        parts.append(missing_md)

    g1 = getattr(r, "gate1_verdict", None)
    g2 = getattr(r, "gate2_verdict", None)
    g1_ok = bool(g1 and getattr(g1, "verdict", "") == "sufficient")
    g2_ok = bool(g2 and getattr(g2, "overall_grounded", False))
    badges = (_gate_badge("Gate 1 coverage", g1_ok) +
              " <span style='color:#888'>-&gt;</span> " +
              _gate_badge("Gate 2 groundedness", g2_ok))
    corrections = getattr(r, "regeneration_count", 0)
    verdict_word = "Partially verified" if outcome == "partial" else "Verified"
    parts.append(badges + f"\n\n_{verdict_word} · {getattr(r,'total_latency_ms',0)/1000:.1f}s · "
                 f"{corrections} self-correction{'' if corrections == 1 else 's'} · "
                 f"~${getattr(r,'estimated_cost_usd',0):.4f}_")

    if SHOW_CANDIDATE and getattr(r, "candidate_answer", None):
        reason = getattr(r, "candidate_reason", None)
        parts.append("### [!] Unverified candidate (not quality-checked)\n\n"
                     + (f"_{reason}_\n\n" if reason else "") + r.candidate_answer)

    return "\n\n".join(parts), _sources_md(getattr(r, "citations", []) or [])


import re as _re


def _clean_table_cell(text: str) -> str:
    """Strip PDF text-extraction artefacts from table content: leader dots
    (". . . ."), merged CamelCase words, and runs of whitespace. Applied only to
    table chunks (the core already serialized them to GFM markdown; we clean that
    serialized string, since per-cell data is not exposed here)."""
    if not isinstance(text, str):
        return str(text)
    text = _re.sub(r"(\.\s*){3,}", " ", text)
    text = _re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = _re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _pdf_link(abs_path: str, page=None) -> str:
    """A clickable link that opens the file (at an exact page for PDFs) in a new
    tab. Gradio 5 serves it at /gradio_api/file=<abs_path> when the dir is in allowed_paths."""
    if not abs_path:
        return ""
    if str(abs_path).lower().endswith(".pdf") and page not in (None, "", "-"):
        return f'<a href="/gradio_api/file={abs_path}#page={page}" target="_blank">p.{page} (open)</a>'
    return f'<a href="/gradio_api/file={abs_path}" target="_blank">open</a>'


def _resolve_doc_path(doc_name: str) -> str | None:
    """Absolute path for a retrieved chunk's document, when resolvable. Demo-corpus
    files resolve by name; session uploads (Gradio temp paths) do not, so they get
    no link rather than a broken one."""
    if not doc_name:
        return None
    cand = H.DEMO_CORPUS_DIR / doc_name
    return str(cand) if cand.exists() else None


def _provenance(content_types) -> str:
    cts = content_types or []
    labels = [n for n, key in (("text", "text"), ("figure", "figure_caption"),
                               ("table", "table")) if key in cts]
    return " + ".join(labels) or "text"


def _sources_md(citations) -> str:
    if not citations:
        return ""
    from pathlib import Path
    media = H.media_by_chunk_id(citations)
    out = ["### Sources"]
    for c in citations:
        prov = _provenance(getattr(c, "content_types", []))
        if getattr(c, "source_type", "") == "external":
            out.append(f"**[{c.number}]** *external* — [{c.url}]({c.url}) · {prov}")
        else:
            src = getattr(c, "source_path", "") or ""
            page_list = [p for p in (getattr(c, "pages", []) or [])]
            pages = ", ".join(str(p) for p in page_list)
            ptxt = f" · page(s) {pages}" if pages else ""
            link = f" · {_pdf_link(src, page_list[0] if page_list else None)}" if src else ""
            out.append(f"**[{c.number}]** *local* — `{src}`{ptxt} · {prov}{link}")
        for cid in getattr(c, "contributing_chunk_ids", []) or []:
            m = media.get(cid)
            if not m:
                continue
            if m["content_type"] == "table" and m.get("text"):
                out.append(f"_Table — page {m.get('page')}_\n\n{_clean_table_cell(m['text'])}")
            elif m["content_type"] == "figure_caption":
                ip = m.get("image_path")
                if ip and os.path.exists(ip) and H.image_path_is_safe(ip):
                    out.append(f"![figure]({ip})\n\n_Figure — page {m.get('page')}_")
                elif m.get("text"):
                    out.append(f"_Figure — page {m.get('page')}_: {m['text']}")
    return "\n\n".join(out)


def _excerpts_md(citations) -> str:
    if not citations:
        return ""
    out = ["### Relevant excerpts from your documents",
           "_The retrieved passages the generator saw. Shown because the answer "
           "could not be verified — inspect them yourself._"]
    for c in citations:
        txt = getattr(c, "parent_text", "") or ""
        out.append(f"> {html.escape(txt[:1200])}")
    return "\n\n".join(out)


def _trace_md(r) -> str:
    trace = getattr(r, "trace", []) or []
    head = (f"_{len(trace)} step(s) · {getattr(r,'regeneration_count',0)} regeneration(s) · "
            f"{getattr(r,'llm_call_count',0)} LLM call(s)_\n")
    lines = [head]
    for i, step in enumerate(trace, 1):
        lines.append(f"{i}. {H.friendly_step(step)}  \n   _{step}_")
    return "\n".join(lines)


def ask(question, pdf_format, corpus_only, free_used, byo_key, session_id):
    """Streaming handler. Yields (status, answer, sources, trace, pipelog_md, free_used,
    missing_facts, query_ctx) -- missing_facts is a MissingFacts on a result, else None;
    query_ctx is {"question", "use_backend"} on a result, else None.

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
        yield (f"[!] {e}", "", "", "", "Run a query to see the pipeline trace.", free_used, None, None)
        return

    visitor_free = free_used < FREE_QUERIES
    use_backend = visitor_free and _budget_allows()
    if not use_backend and not byo_key:
        why = ("The demo's shared daily budget is used up for today."
               if not _budget_allows() else f"You've used your {FREE_QUERIES} free demo queries.")
        yield (f"[key] {why} Paste your own DeepSeek API key below to continue — used "
               "only for your session, never stored.", "", "", "",
               "Run a query to see the pipeline trace.", free_used, None, None)
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
            yield ("[running] Thinking...", "", "", "", "\n\n".join(live_log), free_used, None, None)
        elif kind == "__quota__":
            msg = ("[blocked] **API limit reached on your key.** Wait a moment and try again, or "
                   "use a different key." if payload else
                   "[blocked] **The demo is experiencing high traffic** and has hit its API quota. "
                   "Try again in a few minutes, or enter your own DeepSeek API key.")
            log("quota")
            yield (msg, "", "", "", "\n\n".join(live_log), free_used, None, None)
            return
        elif kind == "__error__":
            log(f"error: {payload}")
            yield (f"[!] Query failed: {payload}", "", "", "", "\n\n".join(live_log), free_used, None, None)
            return
        elif kind == "__result__":
            result = payload

    if result is None:
        log("no result")
        yield ("[!] Query failed: no result produced.", "", "", "",
               "\n\n".join(live_log), free_used, None, None)
        return

    if use_backend:
        _budget_consume()
        free_used += 1
    missing = H.extract_missing_facts(result, question=question, user_key=user_key)
    answer_md, sources_md = _render_result(question, result, missing)
    status = "[OK] Done" + (f" · {FREE_QUERIES - free_used} free queries left" if use_backend else "")
    full_log = _pipelog_md(list(getattr(result, "pipeline_log", None) or []))
    log(f"done: outcome={getattr(result,'outcome',None)}")
    # What the follow-up suggestions step needs to bill/key exactly like this query:
    # which key path was used (never the key itself).
    query_ctx = {"question": question, "use_backend": use_backend}
    yield (status, answer_md, sources_md, _trace_md(result), full_log, free_used, missing,
           query_ctx)


# ---------------------------------------------------------------------------
# Pipeline Log rendering
# ---------------------------------------------------------------------------
_TYPE_COLORS = {"text": "#6495ed", "table": "#2ea043", "image caption": "#ab7dff"}


def _pipelog_md(events) -> str:
    if not events:
        return "Run a query to see the pipeline trace."
    from verirag.pipeline_log import (abstain_parts, gate1_line, gate2_rows,
                                      included_lines, retrieval_rows)

    def of(stage):
        return [e for e in events if e.stage == stage]

    out: list[str] = []
    rets = of("retrieval")
    if rets:
        ev = rets[-1]
        n, m = ev.detail["chunk_count"], ev.detail["doc_count"]
        out.append(f"### Retrieval — {n} chunk(s) from {m} document(s)")
        rows = retrieval_rows(ev.detail["chunks"])
        for row, chunk in zip(rows, ev.detail["chunks"]):
            abs_path = _resolve_doc_path(chunk.get("doc_name"))
            if abs_path:                       # CHANGE 4: clickable page link
                row["Page"] = _pdf_link(abs_path, chunk.get("page"))
            if "table" in str(row.get("Type", "")).lower():   # CHANGE 5: clean table text
                row["Snippet"] = _clean_table_cell(row.get("Snippet", ""))
        out.append(_rows_table(rows,
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
        out.append(f"<details><summary>View exact context — "
                   f"{ev.detail['char_count']:,} characters</summary>\n\n```\n"
                   f"{ev.detail['full_context'][:20000]}\n```\n\n</details>")

    out.append("### Gate 1 — Coverage")
    g1 = of("gate1")
    if g1:
        for ev in g1:
            ok = ev.detail["verdict"] == "pass"
            out.append(("[OK] " if ok else "[!] ") + gate1_line(ev.detail))
    else:
        out.append("_Gate 1 did not run on this path (whole-document summarization skips it)._")

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
        if len(g2) > 1:
            out.append(f"_Gate 2 ran {len(g2)} times; showing the final check._")
        claims = g2[-1].detail["claims"]
        if claims:
            out.append(_rows_table(gate2_rows(claims),
                                   ["Claim", "Verdict", "Source Type", "Score"]))
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


def _rows_table(rows, cols) -> str:
    """A GFM markdown table from a list of dict rows (Gradio Markdown renders it)."""
    if not rows:
        return "_(none)_"
    present = [c for c in cols if any(c in r for r in rows)]
    head = "| " + " | ".join(present) + " |"
    sep = "| " + " | ".join("---" for _ in present) + " |"
    body = []
    for r in rows:
        cells = []
        for c in present:
            v = r.get(c, "")
            if isinstance(v, float):
                v = f"{v:.3f}"
            cells.append(str(v).replace("\n", " ").replace("|", "\\|")[:200])
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


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



# ---------------------------------------------------------------------------
# arXiv suggestions (Feature 3)
# ---------------------------------------------------------------------------
_PAPERS_LABEL = "Papers that might fill the gaps"


def _md_escape(text: str) -> str:
    """Neutralise markdown/HTML in text that comes from arXiv or the LLM."""
    text = str(text or "").replace("\\", "\\\\")
    for ch in "*_`[]#|":
        text = text.replace(ch, "\\" + ch)
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _paper_md(p) -> str:
    return (f"**{_md_escape(p.title)}** ({_md_escape(p.year)}) — {_md_escape(p.authors)}\n\n"
            f"{_md_escape(p.reason)}\n\n"
            f"*{_md_escape(p.abstract_snippet)}...*\n\n"
            f"[View on ArXiv]({p.arxiv_url})")


def _papers_updates(acc, header, papers, status=""):
    """The full output tuple: accordion, header md, N paper mds, N buttons,
    papers state, status md."""
    mds = [gr.update(value=_paper_md(papers[i]), visible=True) if i < len(papers)
           else gr.update(value="", visible=False) for i in range(MAX_PAPER_SUGGESTIONS)]
    btns = [gr.update(visible=i < len(papers), interactive=True)
            for i in range(MAX_PAPER_SUGGESTIONS)]
    return (acc, header, *mds, *btns, list(papers), status)


def _papers_hidden():
    return _papers_updates(gr.update(visible=False, label=_PAPERS_LABEL), "", [])


def papers_flow(missing, query_ctx, byo_key):
    """Runs as a .then() AFTER ask() has rendered the answer, so it never delays
    it. Hidden for a complete answer; otherwise "Searching ArXiv..." first, then
    the suggestions. Keys/budget: reuses the query's own key path (server budget
    or the visitor's key), recorded in query_ctx by ask()."""
    if missing is None or query_ctx is None or getattr(missing, "is_complete", True):
        yield _papers_hidden()
        return
    use_backend = bool(query_ctx.get("use_backend"))
    user_key = None if use_backend else ((byo_key or "").strip() or None)
    if not use_backend and not user_key:
        yield _papers_updates(gr.update(visible=True, label=_PAPERS_LABEL),
                              "Add your DeepSeek API key above to search arXiv for papers.", [])
        return
    yield _papers_updates(gr.update(visible=True, label=f"{_PAPERS_LABEL} (searching...)"),
                          "Searching ArXiv...", [])
    papers = H.suggest_papers(missing, query_ctx.get("question") or "", user_key,
                              max_results=MAX_PAPER_SUGGESTIONS)
    if not papers:
        yield _papers_updates(gr.update(visible=True, label=f"{_PAPERS_LABEL} (none found)"),
                              "No arXiv papers found that clearly cover what your documents "
                              "are missing.", [])
        return
    header = (f"{len(papers)} arXiv paper(s) that may cover what your documents are missing. "
              "Adding one downloads it into your private corpus (this can take a few "
              "minutes for papers with many figures).")
    yield _papers_updates(gr.update(visible=True, label=f"{_PAPERS_LABEL} ({len(papers)})"),
                          header, papers)


def _make_add_paper(i: int):
    def add_paper(papers, session_id):
        if not papers or i >= len(papers):
            yield "[!] That suggestion is no longer available. Ask again.", gr.update()
            return
        p = papers[i]
        title = _md_escape(p.title)
        yield f"[running] Adding **{title}**...", gr.update()
        try:
            _, skipped = H.ingest_from_url(p.pdf_url, session_id, MAX_PDF_MB, MAX_PDF_PAGES)
        except DemoError as e:
            yield f"[!] {e}", gr.update()
            return
        except Exception as e:  # noqa: BLE001
            yield f"[!] {H._friendly(e)}", gr.update()
            return
        msg = (f"[i] **{title}** is already in your corpus. Re-ask your question to use it."
               if skipped else f"[ok] Added **{title}**. Re-ask your question to use it.")
        yield msg, _corpus_md(session_id)
    return add_paper


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="VeriRAG", theme=gr.themes.Soft(), fill_width=True, css=_CSS) as demo:
        # The CALLABLE, not its result: gr.State evaluates a plain value once at
        # build time, so every visitor would share one session id -- and with it
        # each other's uploads. Passing the function makes Gradio call it per
        # visitor, which is what the session-scoped retrieval assumes.
        session_id = gr.State(value=H.new_session_id)
        # Last query's MissingFacts (demo_helpers.MissingFacts or None), for
        # Features 3 (recommendations) and 4 (report) to read without recomputing.
        missing_state = gr.State(value=None)
        # {"question", "use_backend"} of the last answered query (never the key):
        # lets the arXiv step bill exactly like the query it follows.
        query_ctx = gr.State(value=None)
        papers_state = gr.State(value=[])        # list[PaperSuggestion] on screen
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
                gr.Markdown("**Or add by URL**")
                url_box = gr.Textbox(
                    label="Document URL",
                    placeholder="https://arxiv.org/abs/2310.11511",
                    lines=1,
                )
                url_btn = gr.Button("Fetch and ingest", size="sm")
                url_status = gr.Markdown()
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
                        with gr.Accordion(_PAPERS_LABEL, open=False, visible=False) as papers_acc:
                            papers_md = gr.Markdown()
                            paper_mds, paper_btns = [], []
                            for _i in range(MAX_PAPER_SUGGESTIONS):
                                paper_mds.append(gr.Markdown(visible=False))
                                paper_btns.append(gr.Button("Add to corpus", size="sm",
                                                            visible=False))
                            paper_status = gr.Markdown()
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
        url_btn.click(ingest_url, inputs=[url_box, session_id], outputs=url_status).then(
            _corpus_md, inputs=session_id, outputs=corpus_md)
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
            outputs=[status, answer, sources, trace, pipelog_md, free_used, missing_state,
                     query_ctx],
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

        # arXiv suggestions: a separate step AFTER the answer has rendered.
        _papers_out = [papers_acc, papers_md, *paper_mds, *paper_btns, papers_state, paper_status]
        _papers_io = {"inputs": [missing_state, query_ctx, byo_key], "outputs": _papers_out}
        prun = run.then(papers_flow, **_papers_io)
        psrun = srun.then(papers_flow, **_papers_io)
        # A new question clears the old suggestions and cancels a search in flight.
        for trigger in (ask_btn.click, question.submit):
            trigger(_papers_hidden, outputs=_papers_out, cancels=[prun, psrun])
        for _i, _btn in enumerate(paper_btns):
            _btn.click(_make_add_paper(_i), inputs=[papers_state, session_id],
                       outputs=[paper_status, corpus_md])

        # Stop cancels whichever generator is running, resets the status, re-idles.
        stop_btn.click(lambda: "[stopped] Query cancelled.", outputs=status,
                       cancels=[run, srun, prun, psrun]).then(_idle, outputs=[stop_btn, ask_btn])

    return demo


if __name__ == "__main__":
    if not H.server_key_configured():
        print("WARNING: DEEPSEEK_API_KEY is not set — the free path will fail until it is.")
    _DEMO_NAMES = H.startup()
    print(f"Demo corpus preloaded: {_DEMO_NAMES}")
    build_demo().queue(max_size=16).launch(
        server_name="0.0.0.0",
        # serve demo-corpus files (and uploads) over /gradio_api/file= so PDF page links open
        allowed_paths=[str(H.DEMO_CORPUS_DIR), str(H.UPLOAD_TEMP_DIR)],
    )
