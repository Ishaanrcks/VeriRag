"""Natural-language-inference entailment scoring for Gate 2.

Uses a local encoder-only cross-encoder (``cross-encoder/nli-deberta-v3-base``
by default) to score whether a premise entails a hypothesis.

LABEL ORDER IS READ FROM THE CHECKPOINT, NEVER HARDCODED. Index order varies
between NLI checkpoints and guessing wrong silently inverts every verdict --
this model, for instance, reports
``{0: 'contradiction', 1: 'entailment', 2: 'neutral'}``, where entailment sits
at index 1 rather than the index 2 that the common MNLI convention would
suggest. The mapping is built from ``model.config.id2label`` at load time, and
if those labels cannot be interpreted we fail loudly instead of guessing.

``CrossEncoder.predict`` returns raw logits for a multi-class head, so the
softmax is applied here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from sentence_transformers import CrossEncoder

from langsmith import traceable

from verirag.config import settings

logger = logging.getLogger(__name__)

NliLabel = Literal["entailment", "neutral", "contradiction"]

_CANONICAL_LABELS = ("entailment", "neutral", "contradiction")


@dataclass
class EntailmentResult:
    """Softmaxed NLI output for one (premise, hypothesis) pair."""

    label: NliLabel
    # Gate 2 thresholds on this probability, not on the argmax label.
    entailment_prob: float
    all_probs: dict[str, float]


@lru_cache(maxsize=1)
def get_nli_model() -> CrossEncoder:
    """Return the lazily-loaded, process-wide NLI cross-encoder singleton."""
    logger.info("Loading NLI model: %s", settings.nli_model)
    return CrossEncoder(settings.nli_model)


def _canonicalize(raw_label: str) -> str | None:
    """Map a checkpoint's label string onto our canonical three."""
    lowered = raw_label.strip().lower()
    if "entail" in lowered:
        return "entailment"
    if "neutral" in lowered:
        return "neutral"
    if "contradict" in lowered:
        return "contradiction"
    return None


@lru_cache(maxsize=1)
def _index_to_label() -> dict[int, str]:
    """Build {logit_index: canonical_label} from the checkpoint's own config.

    Raises if the config's labels are missing or uninterpretable (e.g. generic
    ``LABEL_0`` names). Failing loudly is deliberate: a wrong mapping would
    invert entailment and neutral without any visible error.
    """
    config = get_nli_model().model.config
    id2label = dict(getattr(config, "id2label", {}) or {})
    if not id2label:
        raise RuntimeError(
            f"NLI model {settings.nli_model!r} exposes no id2label mapping; "
            "cannot determine entailment index without guessing."
        )

    mapping: dict[int, str] = {}
    for index, raw_label in id2label.items():
        canonical = _canonicalize(str(raw_label))
        if canonical is None:
            raise RuntimeError(
                f"NLI model {settings.nli_model!r} reports uninterpretable "
                f"label {raw_label!r} at index {index}. Refusing to guess the "
                "entailment index -- a wrong guess silently inverts verdicts."
            )
        mapping[int(index)] = canonical

    if set(mapping.values()) != set(_CANONICAL_LABELS):
        raise RuntimeError(
            f"NLI model {settings.nli_model!r} does not expose the expected "
            f"three NLI labels; got {sorted(set(mapping.values()))}."
        )

    logger.info("NLI label mapping read from checkpoint config: %s", mapping)
    return mapping


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - np.max(logits)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


@traceable(name="check_entailment")
def check_entailment(premise: str, hypothesis: str) -> EntailmentResult:
    """Score whether ``premise`` entails ``hypothesis``."""
    index_to_label = _index_to_label()

    raw = np.asarray(
        get_nli_model().predict([(premise, hypothesis)]), dtype=np.float64
    )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    probabilities = _softmax(raw[0])

    all_probs = {
        index_to_label[index]: float(probabilities[index])
        for index in range(len(probabilities))
    }
    label = max(all_probs, key=lambda key: all_probs[key])

    return EntailmentResult(
        label=label,  # type: ignore[arg-type]
        entailment_prob=all_probs["entailment"],
        all_probs=all_probs,
    )
