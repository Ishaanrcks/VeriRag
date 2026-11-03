"""RAGAS wiring: a Groq-hosted judge + local BAAI embeddings, no OpenAI key.

Used by both the local smoke test and the Modal run, so the judge, the embeddings
and the metric set are defined once. The judge is openai/gpt-oss-120b on Groq --
the strongest chat model this key can reach (no DeepSeek or Llama are available on
it). gpt-oss is a reasoning model, so max_tokens is set high: a tight cap is spent
entirely on hidden reasoning and leaves the visible judgement empty.
"""
from __future__ import annotations
import os
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset

JUDGE_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def build_judge():
    return LangchainLLMWrapper(ChatOpenAI(
        model=JUDGE_MODEL,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.0,
        max_tokens=2048,          # headroom for a reasoning model's hidden tokens
        timeout=120,
    ))


def build_embeddings(cache_folder: str | None = None):
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        cache_folder=cache_folder,
        encode_kwargs={"normalize_embeddings": True},
    ))


def score_rows(rows: list[dict], cache_folder: str | None = None, max_workers: int = 4):
    """Run the four metrics over rows of {question, answer, contexts, ground_truth}.

    Returns the per-question ragas result. A row with no contexts still scores
    faithfulness/relevancy; context_precision/recall are NaN there, as ragas defines.
    """
    ds = Dataset.from_list([
        dict(question=r["question"], answer=r["answer"],
             contexts=r.get("contexts") or [], ground_truth=str(r["ground_truth"]))
        for r in rows
    ])
    return evaluate(
        ds, metrics=METRICS,
        llm=build_judge(), embeddings=build_embeddings(cache_folder),
        run_config=RunConfig(max_workers=max_workers, timeout=180),
        raise_exceptions=False,
    )
