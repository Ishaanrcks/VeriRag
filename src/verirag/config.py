"""Central configuration for VeriRAG.

Loads settings from environment variables (and a local ``.env`` file) using
pydantic-settings. Import the module-level ``settings`` singleton anywhere in
the codebase rather than constructing ``Settings`` yourself.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings sourced from the environment / ``.env``."""

    # --- API keys / credentials ---
    openai_api_key: str
    anthropic_api_key: str | None = None
    langsmith_api_key: str | None = None
    langsmith_project: str = "verirag"
    tavily_api_key: str | None = None

    # --- Models ---
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # --- Chunking ---
    chunk_min_tokens: int = 150
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # --- Vector store ---
    chroma_persist_dir: str = "./data/chroma"

    # --- Critic / corrective loop ---
    max_critic_iterations: int = 2
    per_query_cost_ceiling_usd: float = 0.50

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
