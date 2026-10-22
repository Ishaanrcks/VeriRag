"""Gate 2: groundedness check, run AFTER generation.

Every factual claim in the answer must be entailed by the passages it cites.
Claims are checked against the SMALL ``raw_text`` of each contributing chunk
rather than ``parent_text``: the NLI encoder has a ~512-token input limit and
parent_text runs up to ``parent_text_max_chars`` (6000), which would be
silently truncated mid-premise. Chunks are already bounded by
``chunk_max_tokens``, so they fit properly.

MODEL INDEPENDENCE: entailment is scored by a local encoder-only model
(DeBERTa, via critic/nli.py), which is a different model family from the
DeepSeek decoder LLM used for generation and claim extraction. The verifier
therefore cannot share the generator's failure modes -- this satisfies the
spec's "different model family" requirement by construction, not by
coincidence.

Like Gate 1, this module only emits a verdict. Acting on it (regeneration) is
Phase 8's LangGraph loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from verirag.config import settings
from verirag.critic.claims import extract_claims
from verirag.critic.nli import check_entailment
from verirag.generation.generator import GeneratedAnswer

logger = logging.getLogger(__name__)

ClaimVerdictLabel = Literal[
    "supported_single",
    "supported_combination",
    "not_supported",
    "absence_unverified",
    "meta_excluded",
]


@dataclass
class ClaimVerdict:
    """How one claim fared against its cited evidence."""

    claim_text: str
    claim_type: str
    verdict: ClaimVerdictLabel
    evidence_chunk_ids: list[str] = field(default_factory=list)
    entailment_prob: float | None = None


@dataclass
class Gate2Verdict:
    """Groundedness outcome for a whole answer."""

    overall_grounded: bool
    claims: list[ClaimVerdict]
    unsupported_claims: list[ClaimVerdict]


def _contributing_chunk_ids(
    cited_numbers: list[int], citation_by_number: dict
) -> list[str]:
    """Flatten the chunk_ids behind a claim's cited citation numbers."""
    chunk_ids: list[str] = []
    for number in cited_numbers:
        citation = citation_by_number.get(number)
        if citation is None:
            logger.warning(
                "Claim cites citation number %s, which the answer does not "
                "define; ignoring it.",
                number,
            )
            continue
        for chunk_id in citation.contributing_chunk_ids:
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
    return chunk_ids


def check_groundedness(answer: GeneratedAnswer) -> Gate2Verdict:
    """Verify every factual claim in ``answer`` against its cited chunks."""
    claims = extract_claims(answer)
    citation_by_number = {c.number: c for c in answer.citations}
    text_by_chunk_id = {h.chunk_id: h.raw_text for h in answer.hits_used}

    verdicts: list[ClaimVerdict] = []

    for claim in claims:
        if claim.claim_type == "meta":
            # Commentary about the retrieved context, not a domain fact.
            # Excluded from scoring entirely; no NLI call.
            verdicts.append(
                ClaimVerdict(
                    claim_text=claim.text,
                    claim_type=claim.claim_type,
                    verdict="meta_excluded",
                )
            )
            continue

        if claim.claim_type == "absence":
            # KNOWN MVP LIMITATION: absence claims are passed through
            # unverified, not checked. Standard NLI entailment cannot confirm a
            # negative -- a passage that simply never discusses a topic neither
            # entails nor contradicts "this source does not mention it", so the
            # score would be meaningless. A production system needs a dedicated
            # check (e.g. confirming the claimed topic really has zero
            # lexical/semantic overlap with the cited source).
            verdicts.append(
                ClaimVerdict(
                    claim_text=claim.text,
                    claim_type=claim.claim_type,
                    verdict="absence_unverified",
                )
            )
            continue

        # --- factual claim: check entailment against its cited chunks ---
        chunk_ids = _contributing_chunk_ids(claim.cited_numbers, citation_by_number)
        resolved = [
            (chunk_id, text_by_chunk_id[chunk_id])
            for chunk_id in chunk_ids
            if chunk_id in text_by_chunk_id
        ]

        if not resolved:
            # An uncited (or undefined-citation) factual claim has no evidence
            # to stand on, so it cannot be grounded.
            logger.warning(
                "Factual claim has no resolvable evidence chunks: %r",
                claim.text[:100],
            )
            verdicts.append(
                ClaimVerdict(
                    claim_text=claim.text,
                    claim_type=claim.claim_type,
                    verdict="not_supported",
                    evidence_chunk_ids=chunk_ids,
                    entailment_prob=None,
                )
            )
            continue

        threshold = settings.gate2_entailment_threshold
        best_prob: float | None = None
        single_support: tuple[str, float] | None = None

        # (b) each contributing chunk on its own first.
        for chunk_id, premise in resolved:
            result = check_entailment(premise, claim.text)
            if best_prob is None or result.entailment_prob > best_prob:
                best_prob = result.entailment_prob
            if result.entailment_prob >= threshold:
                single_support = (chunk_id, result.entailment_prob)
                break

        if single_support is not None:
            verdicts.append(
                ClaimVerdict(
                    claim_text=claim.text,
                    claim_type=claim.claim_type,
                    verdict="supported_single",
                    evidence_chunk_ids=[single_support[0]],
                    entailment_prob=single_support[1],
                )
            )
            continue

        # (c) chunks jointly, for claims that span several passages.
        # Skipped when there is only one chunk: the combined premise would be
        # that same text (or a truncation of it), so the probability cannot
        # exceed what step (b) already measured.
        all_chunk_ids = [chunk_id for chunk_id, _ in resolved]
        if len(resolved) > 1:
            combined = "\n\n".join(text for _, text in resolved)[
                : settings.gate2_combination_max_chars
            ]
            combined_result = check_entailment(combined, claim.text)
            if best_prob is None or combined_result.entailment_prob > best_prob:
                best_prob = combined_result.entailment_prob
            if combined_result.entailment_prob >= threshold:
                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim.text,
                        claim_type=claim.claim_type,
                        verdict="supported_combination",
                        evidence_chunk_ids=all_chunk_ids,
                        entailment_prob=combined_result.entailment_prob,
                    )
                )
                continue

        # (d) nothing cleared the threshold.
        verdicts.append(
            ClaimVerdict(
                claim_text=claim.text,
                claim_type=claim.claim_type,
                verdict="not_supported",
                evidence_chunk_ids=all_chunk_ids,
                entailment_prob=best_prob,
            )
        )

    unsupported = [v for v in verdicts if v.verdict == "not_supported"]
    # absence_unverified and meta_excluded deliberately do NOT count against
    # grounding.
    return Gate2Verdict(
        overall_grounded=not unsupported,
        claims=verdicts,
        unsupported_claims=unsupported,
    )
