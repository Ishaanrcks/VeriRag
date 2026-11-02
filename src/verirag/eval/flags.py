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

FLAGS = (
    "gate1_advisory",
    "gate2_citation_repair",
    "gate2_numeric_match",
    "gate2_visual_verify",
    "refusal_requires_search",
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
    if temperature is not None:
        settings.generation_temperature = temperature
    logger.info("benchmark flags on: %s (generation temperature %s)",
                wanted or "none", settings.generation_temperature)
    return wanted


def active_flags() -> list[str]:
    return [n for n in FLAGS if getattr(settings, n, False)]
