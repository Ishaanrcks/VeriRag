"""Gate 2: groundedness check, run AFTER generation.

Every factual claim in the answer must be entailed by the passages it cites.
Claims are checked against the SMALL ``raw_text`` of each contributing chunk
rather than ``parent_text``: the NLI encoder has a ~512-token input limit and
parent_text runs up to ``parent_text_max_chars`` (6000), which would be
silently truncated mid-premise. Chunks are already bounded by
``chunk_max_tokens``, so they fit properly.

EXTERNAL CORROBORATION POLICY: evidence from the user's own local documents
grounds a claim on its own, unchanged. Evidence consisting ONLY of fetched web
content must be corroborated by at least two DISTINCT source URLs (two chunks
off the same page are one source, not two). A claim resting on a single
unverified web result is reported as ``external_unsupported``, which blocks
grounding just like ``not_supported``.

The reasoning is worth being able to defend rather than treating as an
arbitrary rule: web results vary widely in reliability and recency, so one
page clearing an entailment threshold is weak evidence that a fact is true --
the NLI model only confirms the page *says* it. Independent agreement across
multiple sources, or any agreement with the user's own documents, is a
defensible trust bar for an MVP. It is deliberately asymmetric: we trust the
corpus the user chose over arbitrary pages we fetched for them.

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
from verirag.retrieval.types import SearchHit

logger = logging.getLogger(__name__)

ClaimVerdictLabel = Literal[
    "supported_single",
    "supported_combination",
    "not_supported",
    "external_unsupported",
    "absence_unverified",
    "meta_excluded",
]

# Verdicts that block grounding. "external_unsupported" is reported as its own
# label for transparency but counts against grounding exactly like
# "not_supported", so it triggers regeneration and eventual abstention.
_BLOCKING_VERDICTS = {"not_supported", "external_unsupported"}


def _is_external(chunk_id: str, hit_by_chunk_id: dict[str, SearchHit]) -> bool:
    """True only for web-fetched evidence.

    A missing ``source_type`` -- or a chunk we cannot resolve -- is treated as
    local, matching the convention set in external/fetcher.py.
    """
    hit = hit_by_chunk_id.get(chunk_id)
    return bool(hit and hit.metadata.get("source_type") == "external")


def _distinct_external_urls(
    chunk_ids: list[str], hit_by_chunk_id: dict[str, SearchHit]
) -> set[str]:
    """Distinct external source URLs behind these chunks.

    Keyed on ``source_path`` rather than chunk_id on purpose: two chunks taken
    from the SAME fetched page are one source, not two corroborating ones.
    """
    urls: set[str] = set()
    for chunk_id in chunk_ids:
        hit = hit_by_chunk_id.get(chunk_id)
        if hit and hit.metadata.get("source_type") == "external":
            urls.add(str(hit.metadata.get("source_path") or chunk_id))
    return urls


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
    hit_by_chunk_id = {h.chunk_id: h for h in answer.hits_used}

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
        supporting: list[tuple[str, float]] = []

        # (b) each contributing chunk on its own first. We stop as soon as a
        # LOCAL chunk clears the bar (local evidence is sufficient by itself),
        # but keep scanning while only external chunks have cleared, because
        # the corroboration rule below needs the full set of supporting
        # sources to count distinct URLs.
        for chunk_id, premise in resolved:
            result = check_entailment(premise, claim.text)
            if best_prob is None or result.entailment_prob > best_prob:
                best_prob = result.entailment_prob
            if result.entailment_prob >= threshold:
                supporting.append((chunk_id, result.entailment_prob))
                if not _is_external(chunk_id, hit_by_chunk_id):
                    break

        all_chunk_ids = [chunk_id for chunk_id, _ in resolved]

        if supporting:
            local_supporting = [
                (chunk_id, prob)
                for chunk_id, prob in supporting
                if not _is_external(chunk_id, hit_by_chunk_id)
            ]
            if local_supporting:
                # Any local support grounds the claim outright.
                best_local = max(local_supporting, key=lambda pair: pair[1])
                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim.text,
                        claim_type=claim.claim_type,
                        verdict="supported_single",
                        evidence_chunk_ids=[best_local[0]],
                        entailment_prob=best_local[1],
                    )
                )
                continue

            # External-only support -> corroboration rule applies.
            supporting_ids = [chunk_id for chunk_id, _ in supporting]
            urls = _distinct_external_urls(supporting_ids, hit_by_chunk_id)
            corroborated = len(urls) >= 2
            if not corroborated:
                logger.info(
                    "Claim supported only by %d external source(s); "
                    "needs 2 distinct URLs to ground: %r",
                    len(urls), claim.text[:100],
                )
            verdicts.append(
                ClaimVerdict(
                    claim_text=claim.text,
                    claim_type=claim.claim_type,
                    verdict=(
                        "supported_combination"
                        if corroborated
                        else "external_unsupported"
                    ),
                    evidence_chunk_ids=supporting_ids,
                    entailment_prob=max(prob for _, prob in supporting),
                )
            )
            continue

        # (c) chunks jointly, for claims that span several passages.
        # Skipped when there is only one chunk: the combined premise would be
        # that same text (or a truncation of it), so the probability cannot
        # exceed what step (b) already measured.
        if len(resolved) > 1:
            combined = "\n\n".join(text for _, text in resolved)[
                : settings.gate2_combination_max_chars
            ]
            combined_result = check_entailment(combined, claim.text)
            if best_prob is None or combined_result.entailment_prob > best_prob:
                best_prob = combined_result.entailment_prob
            if combined_result.entailment_prob >= threshold:
                has_local = any(
                    not _is_external(chunk_id, hit_by_chunk_id)
                    for chunk_id in all_chunk_ids
                )
                urls = _distinct_external_urls(all_chunk_ids, hit_by_chunk_id)
                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim.text,
                        claim_type=claim.claim_type,
                        verdict=(
                            "supported_combination"
                            if (has_local or len(urls) >= 2)
                            else "external_unsupported"
                        ),
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

    # Both not_supported and external_unsupported block grounding, so both
    # feed the corrective loop's avoid_claims and the abstention message.
    # absence_unverified and meta_excluded deliberately do NOT count against
    # grounding.
    unsupported = [v for v in verdicts if v.verdict in _BLOCKING_VERDICTS]
    return Gate2Verdict(
        overall_grounded=not unsupported,
        claims=verdicts,
        unsupported_claims=unsupported,
    )
