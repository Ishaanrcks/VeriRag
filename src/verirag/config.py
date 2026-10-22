"""Central configuration for VeriRAG.

Loads settings from environment variables (and a local ``.env`` file) using
pydantic-settings. Import the module-level ``settings`` singleton anywhere in
the codebase rather than constructing ``Settings`` yourself.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings sourced from the environment / ``.env``."""

    # --- LLM provider (DeepSeek via OpenAI-compatible API) ---
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_chat_model: str = "deepseek-chat"
    deepseek_vision_model: str = "deepseek-v4-flash-vision-exp"

    # --- Other credentials ---
    langsmith_api_key: str | None = None
    langsmith_project: str = "verirag"
    tavily_api_key: str | None = None

    # --- Models ---
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # --- Document loading / figures ---
    # Images smaller than this (in either dimension, px) are treated as
    # logos/icons/bullets and skipped rather than captioned.
    min_figure_dimension_px: int = 150
    # Parent ("big") text attached to each chunk for small-to-big generation,
    # truncated to this many characters.
    parent_text_max_chars: int = 6000

    # --- Chunking ---
    chunk_min_tokens: int = 150
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # --- Vector store (dense) ---
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "verirag_chunks"

    # --- Sparse index (BM25) ---
    bm25_corpus_path: str = "./data/processed/bm25_corpus.jsonl"

    # --- Hybrid retrieval ---
    retrieval_dense_top_k: int = 20
    retrieval_sparse_top_k: int = 20
    retrieval_rrf_k: int = 60
    # Cap the candidate set handed to the cross-encoder; reranking is the
    # expensive stage on CPU, so this bounds latency.
    retrieval_rerank_candidate_cap: int = 20
    retrieval_final_top_k: int = 5

    # --- Critic: Gate 1 (coverage) ---
    # PLACEHOLDER thresholds for the MVP build. These get properly calibrated
    # in Phase 12 against the labeled critic-calibration set -- deliberately
    # NOT tuned by eyeballing fixture data now.
    gate1_high_threshold: float = 0.5
    gate1_low_threshold: float = 0.05

    # --- Critic: Gate 2 (groundedness) ---
    # Also a PLACEHOLDER, calibrated in Phase 12 alongside the Gate 1 bands.
    gate2_entailment_threshold: float = 0.5
    # Cap on the combined premise text when testing whether several chunks
    # jointly support a claim (the NLI model has a ~512-token input limit).
    gate2_combination_max_chars: int = 2000

    # --- Critic / corrective loop ---
    max_critic_iterations: int = 2
    per_query_cost_ceiling_usd: float = 0.50
    # Loose call-count proxy for the cost ceiling, used as a safety bound on
    # the graph's loops. Precise token-based dollar cost tracking is Phase
    # 10's job; this just guarantees a query cannot spin indefinitely.
    max_llm_calls_per_query: int = 10

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
