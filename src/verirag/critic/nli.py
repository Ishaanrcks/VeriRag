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
import re
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


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_premise(text: str) -> str:
    """Clean PDF-extraction noise that degrades NLI scoring.

    Chunks lifted from academic PDFs carry hyphenated line breaks
    ("incorpor-\\nate"), hard newlines mid-sentence, and ligatures. Left in,
    they push an already out-of-distribution premise further from the short,
    clean sentence pairs the checkpoint was trained on.
    """
    cleaned = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    cleaned = re.sub(r"-\s*\n\s*", "", cleaned)  # rejoin hyphenated breaks
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def premise_windows(
    premise: str, window_sentences: int, max_windows: int
) -> list[str]:
    """Split a premise into overlapping sentence windows, best-effort.

    Returns the windows a claim should be scored against. Always includes at
    least one window so a short premise still gets checked.
    """
    normalized = normalize_premise(premise)
    if not normalized:
        return []

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(normalized) if s.strip()]
    if len(sentences) <= window_sentences:
        return [normalized]

    # The whole premise is always a candidate alongside the windows. Without
    # it, windowing could SCORE WORSE than the unwindowed premise for a claim
    # whose support is spread across more sentences than one window spans
    # (measured: a true claim scored 0.9558 on the full chunk but only 0.1840
    # on its best 3-sentence window). Including it makes the windowed result
    # greater than or equal to the old behaviour by construction.
    windows: list[str] = [normalized]
    for start in range(len(sentences) - window_sentences + 1):
        windows.append(" ".join(sentences[start : start + window_sentences]))
        if len(windows) > max_windows:
            break
    return windows


def check_entailment_windowed(
    premise: str,
    hypothesis: str,
    window_sentences: int | None = None,
    max_windows: int | None = None,
) -> EntailmentResult:
    """Score a hypothesis against the BEST sentence window of a premise.

    A long premise dilutes the signal: the relevant sentence is averaged in
    with a dozen irrelevant ones and the model answers "neutral". Scoring
    windows separately and keeping the strongest result recovers the signal,
    and all windows go through the model in one batched call so this stays
    cheaper than it looks.
    """
    windows = premise_windows(
        premise,
        window_sentences
        if window_sentences is not None
        else settings.gate2_premise_window_sentences,
        max_windows
        if max_windows is not None
        else settings.gate2_max_windows_per_chunk,
    )
    if not windows:
        return EntailmentResult(
            label="neutral",
            entailment_prob=0.0,
            all_probs={"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0},
        )

    index_to_label = _index_to_label()
    raw = np.asarray(
        get_nli_model().predict([(window, hypothesis) for window in windows]),
        dtype=np.float64,
    )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    best: EntailmentResult | None = None
    for row in raw:
        probabilities = _softmax(row)
        all_probs = {
            index_to_label[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }
        candidate = EntailmentResult(
            label=max(all_probs, key=lambda key: all_probs[key]),  # type: ignore[arg-type]
            entailment_prob=all_probs["entailment"],
            all_probs=all_probs,
        )
        # "Best" means most entailed; ties on entailment break toward the
        # window that is most confidently contradicting, so an actual
        # contradiction is not hidden behind a neutral window.
        if (
            best is None
            or candidate.entailment_prob > best.entailment_prob
            or (
                candidate.entailment_prob == best.entailment_prob
                and candidate.all_probs["contradiction"]
                > best.all_probs["contradiction"]
            )
        ):
            best = candidate

    assert best is not None
    return best


def _neutral_result() -> EntailmentResult:
    return EntailmentResult(
        label="neutral",
        entailment_prob=0.0,
        all_probs={"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0},
    )


def score_claim_against_premises(
    hypothesis: str, premises: list[str]
) -> list[EntailmentResult]:
    """Score one claim against several premises in ONE batched model call.

    Every sentence window of every premise goes through the cross-encoder
    together, and the best-entailing window per premise is returned, aligned
    with ``premises``. Windowing multiplies the number of forward passes by
    roughly the window count, so batching across premises (rather than one
    call per chunk) is what keeps Gate 2's latency manageable.

    NOTE on contradiction: each result carries its best window's full
    probabilities, but the maximum contradiction ACROSS windows is not
    reported, and must not be used as a hallucination signal. Measured on this
    corpus, true claims reach contradiction 0.998 simply because some
    unrelated sentence in the same chunk contradicts them.
    """
    if not premises:
        return []

    pairs: list[tuple[str, str]] = []
    owners: list[int] = []
    for index, premise in enumerate(premises):
        windows = premise_windows(
            premise,
            settings.gate2_premise_window_sentences,
            settings.gate2_max_windows_per_chunk,
        )
        for window in windows:
            pairs.append((window, hypothesis))
            owners.append(index)

    if not pairs:
        return [_neutral_result() for _ in premises]

    index_to_label = _index_to_label()
    raw = np.asarray(get_nli_model().predict(pairs), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    best: list[EntailmentResult | None] = [None] * len(premises)
    for row, owner in zip(raw, owners):
        probabilities = _softmax(row)
        all_probs = {
            index_to_label[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }
        candidate = EntailmentResult(
            label=max(all_probs, key=lambda key: all_probs[key]),  # type: ignore[arg-type]
            entailment_prob=all_probs["entailment"],
            all_probs=all_probs,
        )
        current = best[owner]
        if current is None or candidate.entailment_prob > current.entailment_prob:
            best[owner] = candidate

    return [result or _neutral_result() for result in best]


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
