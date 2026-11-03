"""Benchmark-only experiment flags.

Each flag names a behaviour change that a benchmark run may switch on. They all
default False in ``config.Settings``, so production runs exactly as it did before
any of them existed; only a benchmark run sets them, and only for its own process.

    apply_flags("gate1_advisory,gate2_numeric_match")

Unknown names raise, so a typo in a run definition fails loudly instead of
silently measuring the baseline twice.
"""

from __future__ import annotations

import logging

from verirag.config import settings

logger = logging.getLogger(__name__)

# Widened retrieval values for the retrieval_wide flag. dense/sparse are raised
# too: the reranker can only see what fusion hands it, so a cap of 40 over a
# 20+20 candidate pool would rarely bind.
WIDE_RETRIEVAL = dict(retrieval_final_top_k=20, retrieval_rerank_candidate_cap=40,
                      retrieval_dense_top_k=40, retrieval_sparse_top_k=40)
NARROW_RETRIEVAL = dict(retrieval_final_top_k=8, retrieval_rerank_candidate_cap=20,
                        retrieval_dense_top_k=20, retrieval_sparse_top_k=20)

FLAGS = (
    "gate1_advisory",
    "gate2_citation_repair",
    "gate2_numeric_match",
    "gate2_visual_verify",
    "refusal_requires_search",
    "retrieval_wide",
    "extractor_format_hint",
    "critic_iterations_one",
    "visual_count",
    "p3_bundle",
    "unverified_candidate",
)


def apply_flags(names: str | list[str], *, temperature: float | None = 0.0) -> list[str]:
    """Turn the named flags on for this process; return the ones applied.

    ``temperature`` pins generation (0 for a benchmark run, so a rerun is
    comparable); pass None to leave the configured value alone.
    """
    wanted = [n.strip() for n in (names.split(",") if isinstance(names, str) else names) if n.strip()]
    unknown = [n for n in wanted if n not in FLAGS]
    if unknown:
        raise ValueError(f"unknown benchmark flag(s) {unknown}; known flags are {list(FLAGS)}")
    for name in FLAGS:
        setattr(settings, name, name in wanted)
    # retrieval_wide is a set of numbers, not a branch in the retrieval code, so it
    # is applied here and explicitly reverted when absent.
    for key, value in (WIDE_RETRIEVAL if "retrieval_wide" in wanted else NARROW_RETRIEVAL).items():
        setattr(settings, key, value)
    # Also a number rather than a branch, so applied and reverted here.
    settings.max_critic_iterations = 1 if "critic_iterations_one" in wanted else 2
    if temperature is not None:
        settings.generation_temperature = temperature
    logger.info("benchmark flags on: %s (generation temperature %s)",
                wanted or "none", settings.generation_temperature)
    return wanted


def active_flags() -> list[str]:
    return [n for n in FLAGS if getattr(settings, n, False)]
