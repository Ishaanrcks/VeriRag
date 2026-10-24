# Deploying VeriRAG to Hugging Face Spaces

## Secrets to set in HF Spaces → Settings → Repository secrets:

| Secret name            | What it is                                     |
|------------------------|------------------------------------------------|
| DEEPSEEK_API_KEY       | Your rotated DeepSeek key                      |
| LANGSMITH_API_KEY      | Optional — omit to disable tracing             |
| TAVILY_API_KEY         | Optional — needed only if web_search tool enabled |
| DEMO_MODE              | Set to "true" for the public Space             |
| MAX_QUERIES_PER_SESSION | Set to "15" (or adjust)                        |

## First-time corpus setup (maintainer only):

1. Clone the Space repo locally.
2. Copy the 5 demo PDFs into demo_corpus/.
3. Run: `python scripts/ingest_demo.py`
4. `git add data/chroma data/processed` (only if you want to commit the
   pre-built index — alternative is to run ingest on Space startup).
5. Push. The Space will serve queries immediately without a cold-ingest wait.

Note: data/chroma is gitignored by default. To commit the pre-built index,
temporarily remove data/chroma from .gitignore for this one commit, then
restore it. Label that commit clearly: "chore: commit pre-built demo index".
