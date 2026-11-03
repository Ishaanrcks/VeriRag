"""Central configuration for VeriRAG.

Loads settings from environment variables (and a local ``.env`` file) using
pydantic-settings. Import the module-level ``settings`` singleton anywhere in
the codebase rather than constructing ``Settings`` yourself.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

# Session id stamped on pre-ingested demo/corpus content, visible to every
# session. A user's own uploads are stamped with their Streamlit session id
# instead and are visible only within that session. Defined here (a light,
# universally-imported module) so the store, sparse index, ingest and UI can
# all share one definition without a heavy cross-import.
SHARED_SESSION_ID = "__shared__"


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
    # MMLongBench-Doc answer extraction (eval only). Gemini is reached through
    # its OpenAI-compatible endpoint, so the same client class serves it.
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # gemini-2.0-flash is retired (404 from the API); 2.5-flash is its GA successor.
    gemini_extraction_model: str = "gemini-2.5-flash"

    # --- Models ---
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # --- Document loading / figures ---
    # Images smaller than this (in either dimension, px) are treated as
    # logos/icons/bullets and skipped rather than captioned.
    min_figure_dimension_px: int = 150
    # Where raw figure image bytes extracted from PDFs are persisted, so the
    # UI can display the real figure above its caption. Sits alongside the
    # other ./data stores (chroma, processed) and is relative to the project
    # root, matching chroma_persist_dir's convention.
    images_dir: str = "./data/images"
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

    # --- External fallback (web search) ---
    external_max_results: int = 3
    tavily_search_depth: str = "basic"
    # Multiplicative discount applied to external hits' rerank_score so
    # untrusted web content does not outrank the user's own documents when
    # both are relevant. Discounts only -- never excludes.
    external_source_score_penalty: float = 0.9

    # --- Hybrid retrieval ---
    retrieval_dense_top_k: int = 20
    retrieval_sparse_top_k: int = 20
    retrieval_rrf_k: int = 60
    # Cap the candidate set handed to the cross-encoder; reranking is the
    # expensive stage on CPU, so this bounds latency.
    retrieval_rerank_candidate_cap: int = 20
    retrieval_final_top_k: int = 8
    # No single source document may contribute more than this many chunks to
    # the final set, so a multi-paper corpus (or two copies of one paper) does
    # not let one document crowd out the rest. Backfilled if the cap cannot
    # fill retrieval_final_top_k.
    retrieval_max_chunks_per_doc: int = 4
    # The best-scoring hits taken before any per-document cap applies, so
    # single-document questions never lose their most relevant chunks to
    # diversity. Diversity is spent on the slots beyond this.
    retrieval_guaranteed_top_k: int = 5

    # --- Critic: Gate 1 (coverage) ---
    # Gate 1 compares max_rerank_score against these two numbers directly; there
    # is still no fitted decision function behind them (the architecture doc
    # specifies a logistic regression over three features reported as a
    # risk-coverage curve; that is NOT what is implemented).
    #
    # high_threshold was 0.5 and is now 0.15, from the tier-1 MMLongBench-Doc run
    # (eval_results/.../tier1_2026-10-31.json, 36 questions). In the escalation
    # band the LLM judge declined 9 questions and was WRONG on 8 of them: the
    # annotated evidence page had actually been retrieved for 7 of those 8, so the
    # judge was refusing evidence it had. Lowering the ceiling to 0.15 skips the
    # judge for the 5 clearest of them (scores 0.169-0.383).
    #
    # Why not lower still: the only unanswerable question the judge correctly
    # declined scored 0.052, and the lowest wrongly-declined answerable one scored
    # 0.056. Freeing all 8 therefore needs ~0.055, which leaves the judge a
    # 0.05-0.055 band -- effectively deleting it -- on a 0.004 margin fitted to one
    # 36-question sample. 0.15 keeps a real judge band and ~3x that margin.
    # Re-fit this on the full 200-question subset before trusting it further.
    gate1_high_threshold: float = 0.15
    gate1_low_threshold: float = 0.05

    # --- Benchmark-only experiment flags (ALL default False: production behaviour
    # is exactly what it was before these existed). eval/flags.py turns them on for
    # one benchmark run; nothing in production sets them.
    gate1_advisory: bool = False
    gate2_citation_repair: bool = False
    gate2_numeric_match: bool = False
    gate2_visual_verify: bool = False
    refusal_requires_search: bool = False
    # f_k: widen retrieval. Measured cause: 8 of the 12 questions gates-off wins
    # and the gated baseline loses had the value outside the top-8 or below the
    # Gate 1 floor. Sets final k 8->20 and the rerank candidate cap 20->40.
    retrieval_wide: bool = False
    # f_fmt: tell the extractor the dataset's answer_format. BENCHMARK ONLY --
    # a real user does not supply it, so this measures an upper bound, not a
    # shippable change.
    extractor_format_hint: bool = False
    # Cap the critic's regeneration loop at 1 instead of 2. Measured on W0: 51% of
    # questions exhausted 2 regenerations and consumed 79% of the runtime (86.6 of
    # 110 minutes) to make 3 answerable questions correct, and a question that
    # regenerated twice had a LOWER hit rate (18%) than one that never regenerated
    # (26%). Benchmark-only; production keeps 2.
    critic_iterations_one: bool = False
    # visual_count: ask the figure itself to count, with two differently-worded
    # blind reads of two different images that must agree. Targets the 15 tier-2
    # questions whose answer is only visible in a picture (13 of which are in no
    # stored chunk at all, because a caption describes a figure rather than
    # enumerating its contents).
    # p3_bundle: three narrow fixes measured on the C2 baseline --
    #   caption_slots   reserve 2 of the 8 final slots for same-document captions
    #                   on questions that mention a figure (Q4: its gold value is
    #                   in a caption ranked #1 that never reached the generator)
    #   list_row_scope  take a list from the row the question names (Q155 built its
    #                   list from the wrong table)
    #   fullest_form    prefer the longest form of an extracted value present in the
    #                   answer (Q186 gave city names where gold wanted full addresses)
    p3_bundle: bool = False
    caption_slots_reserved: int = 2
    visual_count: bool = False
    visual_count_calls_per_question: int = 6
    # Benchmark runs pin generation to temperature 0 so a rerun is comparable;
    # production keeps 0.2.
    generation_temperature: float = 0.2

    # --- Critic: Gate 2 visual verification of caption-only claims ---
    # A claim supported only by a figure caption is not certified by that caption
    # (one model vouching for another's output). Instead the FIGURE is shown to the
    # vision model with the claim and a strict YES/NO/CANNOT_TELL question; only YES
    # certifies. Bounded per answer because each check is a vision call.
    gate2_visual_check_enabled: bool = True
    gate2_visual_checks_per_answer: int = 6
    # The vision model is a reasoning model: hidden reasoning counts against this,
    # and the visible answer is one word.
    gate2_visual_check_max_tokens: int = 2048

    # --- Critic: Gate 2 (groundedness) ---
    # Also HAND-SET, for the same reason as the Gate 1 bands above.
    gate2_entailment_threshold: float = 0.5
    # Cap on the combined premise text when testing whether several chunks
    # jointly support a claim (the NLI model has a ~512-token input limit).
    gate2_combination_max_chars: int = 2000

    # Premise windowing. The NLI cross-encoder is trained on sentence pairs,
    # so a full ~2000-char chunk swamps the signal and collapses to "neutral"
    # for true and false claims alike. Measured on the QueensLand corpus: the
    # same true claim scored entailment 0.0042 against a 2068-char chunk and
    # 0.6769 against a 400-char window of it. Claims are therefore scored
    # against sentence windows within each chunk, taking the best window.
    gate2_premise_window_sentences: int = 3
    gate2_max_windows_per_chunk: int = 8

    # Aggregation stays STRICT: every scoreable claim must be supported. A
    # fractional tolerance was considered and deliberately not adopted -- with
    # the premise bug fixed, windowing alone recovered recall (2/6 -> 4/6 on a
    # labeled battery) with zero false positives, so relaxing the rule was not
    # needed to fix the false negatives and would have made the tolerance the
    # only precision guard. A contradiction-based hard-fail was also rejected:
    # measured, TRUE claims reach contradiction 0.998 because an irrelevant
    # window inside the same chunk "contradicts" them, so the signal cannot be
    # trusted with this checkpoint. Revisit both only with measurement.
    # Fraction of chars of a claim's tokens that must appear in the evidence
    # for an unscorable claim (formula / figure reference) to count as present.
    gate2_presence_overlap_fraction: float = 0.7

    # --- Whole-document summarization ---
    # If a document's full ordered text fits in this many characters it is
    # summarized in a single call; beyond it, map-reduce over ordered batches.
    summarization_context_max_chars: int = 24000
    summarization_batch_max_chars: int = 12000

    # --- Observability ---
    metrics_db_path: str = "./data/metrics.db"
    # Rough MVP placeholder for the per-query cost estimate. This should be
    # checked against DeepSeek's actual current API pricing rather than
    # trusted as precise; real token-level accounting is out of scope here.
    estimated_cost_per_llm_call_usd: float = 0.002

    # --- Critic / corrective loop ---
    max_critic_iterations: int = 2
    per_query_cost_ceiling_usd: float = 0.50
    # Loose call-count proxy for the cost ceiling, used as a safety bound on
    # the graph's loops. Precise token-based dollar cost tracking is Phase
    # 10's job; this just guarantees a query cannot spin indefinitely.
    max_llm_calls_per_query: int = 10

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()


def configured_secret(value: str | None) -> str | None:
    """Return a secret only when it looks like a real value.

    ``.env`` files are routinely copied from ``.env.example``, which leaves
    literal placeholders such as ``tvly-...`` or ``lsv2_...`` behind. Those are
    truthy, so a plain truthiness check would treat them as configured and the
    failure would surface later as a confusing auth error instead of an honest
    "not configured". Shared by the Tavily fetcher and LangSmith tracing.
    """
    cleaned = (value or "").strip()
    if not cleaned or "..." in cleaned:
        return None
    return cleaned
