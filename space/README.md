---
title: VeriRAG Demo
emoji: 🔍
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# VeriRAG — self-verifying document QA (demo)

Upload a PDF, ask a question. VeriRAG answers **only** what it can ground in the
document (two-gate critic + visual verification of figure claims) and abstains
otherwise. This Space runs the pipeline exactly as it ships (the `final-c2` config).

## Deploying this Space

The Space needs the `verirag` package source available to `app.py`. Two options:

1. **Copy the repo into the Space** (simplest): place this `space/` directory's
   files (`app.py`, `demo_helpers.py`, `requirements.txt`, this README) at the
   Space root, and copy the repo's `src/` directory alongside them so the layout is:
   ```
   app.py
   demo_helpers.py
   requirements.txt
   src/verirag/...
   ```
   `demo_helpers.py` adds `../src` and `./src` to `sys.path` automatically.

2. **Install verirag from git** by adding it to `requirements.txt`
   (`pip install git+<repo-url>`), if the repo is reachable.

## Secrets

Set one Space secret:

* `DEEPSEEK_API_KEY` — the server key for the free-query path.

Optional env vars (no code change; read by the pipeline's pydantic settings):

* `UNVERIFIED_CANDIDATE=true` — also surface the separate unverified-candidate
  answer. Off in the frozen config, so by default the candidate panel stays empty.

## Demo safety (all configurable at the top of `app.py`)

* **Per-visitor free queries** (default 5) via `gr.BrowserState` (localStorage).
  Trivially bypassable by clearing site data; a courtesy throttle, not security.
* **Global daily budget** on the server key (default 200/day) — the real spend
  protection. Resets at midnight and on Space restart (see the comment in `app.py`
  for how to make it persistent).
* **No persistence**: each upload is ingested into a throwaway per-session index in
  a temp dir and deleted after the query. Nothing survives a restart or is visible
  to another visitor.
* **Input guards**: ≤10 MB, ≤30 pages, ≤500-char question, friendly errors.

First request downloads the local models (bge embeddings, bge reranker, DeBERTa
NLI, ~0.5–1 GB) — expect a slow first query on a cold CPU Space.
