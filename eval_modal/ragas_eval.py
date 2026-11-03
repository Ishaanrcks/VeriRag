"""RAGAS scoring of the stored C2 / R2 / M0 answers, sharded, merged in-image.

    modal run --detach eval_modal/ragas_eval.py --config C2

Judge: openai/gpt-oss-120b on Groq (the strongest chat model this key reaches; no
DeepSeek or Llama are available on it). Embeddings: BAAI/bge-base-en-v1.5, baked
into the image and run on CPU, so no OpenAI key is needed anywhere.

The dataset is built from the stored benchmark payloads: question, ground truth,
the VERIFIED answer only (never unverified_candidate), and the retrieved contexts
fetched from the Chroma store by the citation chunk_ids. The merge runs in a Modal
function, not the local entrypoint -- the local machine has no ragas and no store,
and doing the merge locally is the mistake that lost four benchmark runs earlier.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal

from eval_modal.image import MODEL_CACHE, VOLUME_PATH, secret as verirag_secret, volume, volumes

APP_PREFIX = "veri-rag"
STORE = f"{VOLUME_PATH}/mmlongbench"
SHARDS = 4
LOCAL_PAYLOADS = "/tmp/claude-501/matrix"   # payloads_{C2,R2,M0}.json live here

# ragas needs the langchain 0.3 family; pinned together so the resolve stays consistent.
RAGAS_PINS = [
    "ragas==0.2.15", "langchain==0.3.14", "langchain-core==0.3.29",
    "langchain-community==0.3.14", "langchain-openai==0.2.14",
    "langchain-text-splitters==0.3.5", "litellm", "sentence-transformers",
    "chromadb", "datasets", "huggingface_hub",
]
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
_bake = (
    "from huggingface_hub import snapshot_download; "
    f"snapshot_download('{EMBED_MODEL}')"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*RAGAS_PINS)
    .env({"HF_HOME": MODEL_CACHE, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .run_commands(f'python -c "{_bake}"')
    .add_local_dir(LOCAL_PAYLOADS, "/payloads")
)

# groq-api-key holds GROQ_API_KEY; verirag_secret carries nothing ragas needs but
# keeps parity with the benchmark app's wiring.
groq_secret = modal.Secret.from_name("groq-api-key", required_keys=["GROQ_API_KEY"])

app = modal.App(f"{APP_PREFIX}-ragas", image=image)

JUDGE_MODEL = "openai/gpt-oss-120b"


def _verified_answer(r: dict) -> str:
    if r.get("answer") and not r.get("abstained"):
        return r["answer"]
    reason = r.get("abstention_reason") or "the retrieved documents do not cover it"
    return f"I cannot answer this question from the provided documents. {reason}"


def _citation_chunk_ids(r: dict) -> list[str]:
    return [cid for c in (r.get("citations") or []) for cid in (c.get("chunk_ids") or [])]


@app.function(cpu=4, memory=8192, timeout=7200, retries=0,
              volumes=volumes, secrets=[verirag_secret(), groq_secret])
def score_shard(config: str, index: int) -> list[dict]:
    """Score one strided shard; returns per-question metric rows."""
    volume.reload()
    import chromadb
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)
    from ragas.run_config import RunConfig
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    records = json.loads(Path(f"/payloads/payloads_{config}.json").read_text())
    questions = [q for shard in records for q in shard["questions"]]
    questions.sort(key=lambda q: q["id"])
    mine = questions[index - 1::SHARDS]

    # real retrieved contexts: the citation chunk_ids' text from the store
    client = chromadb.PersistentClient(path=f"{STORE}/chroma")
    col = client.get_collection(client.list_collections()[0].name)
    rows = []
    for q in mine:
        ids = _citation_chunk_ids(q)
        ctx = []
        if ids:
            got = col.get(ids=ids, include=["documents"])
            ctx = [d for d in got["documents"] if d and d.strip()]
        rows.append(dict(id=q["id"], group=q["group"], expected=q["expected_verdict"],
                         question=q["question"], ground_truth=str(q["ground_truth"]),
                         answer=_verified_answer(q), contexts=ctx))

    ds = Dataset.from_list([dict(question=r["question"], answer=r["answer"],
                                 contexts=r["contexts"], ground_truth=r["ground_truth"])
                            for r in rows])
    judge = LangchainLLMWrapper(ChatOpenAI(
        model=JUDGE_MODEL, base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"], temperature=0.0, max_tokens=2048, timeout=120))
    embed = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True}))
    result = evaluate(
        ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge, embeddings=embed,
        run_config=RunConfig(max_workers=4, timeout=180), raise_exceptions=False)
    df = result.to_pandas()
    out = []
    for i, r in enumerate(rows):
        out.append(dict(id=r["id"], group=r["group"], expected=r["expected"],
                        n_contexts=len(r["contexts"]),
                        faithfulness=_num(df.iloc[i]["faithfulness"]),
                        answer_relevancy=_num(df.iloc[i]["answer_relevancy"]),
                        context_precision=_num(df.iloc[i]["context_precision"]),
                        context_recall=_num(df.iloc[i]["context_recall"])))
    return out


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


@app.function(cpu=2, memory=4096, timeout=1800, volumes=volumes, secrets=[verirag_secret()])
def merge(config: str, shard_rows: list[list[dict]]) -> dict:
    """One report from the shard rows: mean and standard error per metric, in-image."""
    rows = [r for shard in shard_rows for r in shard]
    rows.sort(key=lambda r: r["id"])
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in rows if r[m] is not None]
        n = len(vals)
        mean = sum(vals) / n if n else None
        if n > 1:
            var = sum((x - mean) ** 2 for x in vals) / (n - 1)
            se = (var ** 0.5) / (n ** 0.5)
        else:
            se = None
        summary[m] = dict(mean=mean, se=se, n=n)
    return dict(config=config, n_questions=len(rows), metrics=summary, per_question=rows)


@app.local_entrypoint()
def main(config: str):
    rows = list(score_shard.starmap([(config, i) for i in range(1, SHARDS + 1)]))
    out_dir = Path("/tmp/ragas/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"raw_{config}.json").write_text(json.dumps(rows))   # durable before merge
    summary = merge.remote(config, rows)
    (out_dir / f"ragas_{config}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nRAGAS — {config}  (n={summary['n_questions']}, judge={JUDGE_MODEL})")
    for m, s in summary["metrics"].items():
        mean = f"{s['mean']:.3f}" if s["mean"] is not None else "n/a"
        se = f"{s['se']:.3f}" if s["se"] is not None else "n/a"
        print(f"  {m:<18} mean={mean}  SE={se}  (n={s['n']})")
    print(f"written {out_dir}/ragas_{config}.json")
