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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from langsmith import traceable

from verirag.config import settings
from verirag.critic.claims import extract_claims
from verirag.critic.nli import (
    check_entailment_windowed,
    normalize_premise,
    score_claim_against_premises,
)
from verirag.generation.generator import GeneratedAnswer
from verirag.retrieval.types import SearchHit

logger = logging.getLogger(__name__)

ClaimVerdictLabel = Literal[
    "supported_single",
    "supported_combination",
    "supported_presence",
    "caption_only_uncertified",
    "not_supported",
    "external_unsupported",
    "absence_unverified",
    "false_hedge",
    "meta_excluded",
]

_TOKEN_RE = re.compile(r"\w{3,}")
# Words that carry no evidential weight, so requiring them in the evidence
# would only add noise to the presence check.
_PRESENCE_STOPWORDS = frozenset(
    """the and for that with this from are was were has have had its their
    paper document passage states state describes describe shows show
    reports report figure table section appendix equation which while
    where when what how many does did not only also been being""".split()
)


# Provenance tier 1: content extracted directly from the source document, so
# it may certify a claim on its own. "text" is page text; "table" is a table
# serialized from the document's own structure -- both are document-derived,
# not model output. A "figure_caption" is the vision model's description of a
# figure, so it is NOT here: certifying against it would mean one model's
# output vouching for another's.
CERTIFIABLE_CONTENT_TYPES = {"text", "table"}


def _is_model_generated(chunk_id: str, hit_by_chunk_id: dict[str, SearchHit]) -> bool:
    """True for chunks this system generated (vision captions), i.e. not
    directly extracted from the document and so unable to certify alone.

    A chunk is certifiable when its ``content_type`` is in
    :data:`CERTIFIABLE_CONTENT_TYPES` ("text" or "table"). A missing hit or an
    absent content_type is treated as certifiable (the prior default), so this
    only ever withholds certification from an explicitly model-generated type.
    """
    hit = hit_by_chunk_id.get(chunk_id)
    if not hit:
        return False
    content_type = hit.metadata.get("content_type")
    return content_type is not None and content_type not in CERTIFIABLE_CONTENT_TYPES


def _figure_pointer(
    chunk_ids: list[str], hit_by_chunk_id: dict[str, SearchHit]
) -> str:
    """Human-readable "look at this figure" pointer for an uncertified claim."""
    parts: list[str] = []
    for chunk_id in chunk_ids:
        hit = hit_by_chunk_id.get(chunk_id)
        if not hit:
            continue
        page = hit.metadata.get("page")
        source = Path(str(hit.metadata.get("source_path") or "")).name or "source"
        where = f"page {page}" if page is not None else "an unnumbered page"
        part = f"a figure on {where} of {source}"
        if part not in parts:
            parts.append(part)
    return "addressed in " + (" and ".join(parts) if parts else "a figure")


def all_chunk_ids_for(resolved: list[tuple[str, str]]) -> list[str]:
    return [chunk_id for chunk_id, _ in resolved]


def _presence_supported(claim_text: str, premises: list[str]) -> tuple[bool, float]:
    """Token-overlap check for claims NLI cannot score.

    Formulas ("H = GNN(X, E)") and figure/table references are not
    propositions, so entailment is meaningless for them -- the checkpoint
    expects a declarative sentence pair. Verifying that the claim's
    distinctive tokens actually occur in the cited evidence is cheap and is
    the right question for this claim type: does the source really contain
    this artifact?
    """
    tokens = {
        token.lower()
        for token in _TOKEN_RE.findall(claim_text)
        if token.lower() not in _PRESENCE_STOPWORDS
    }
    if not tokens:
        return False, 0.0

    evidence = normalize_premise(" ".join(premises)).lower()
    evidence_tokens = set(_TOKEN_RE.findall(evidence))
    overlap = len(tokens & evidence_tokens) / len(tokens)
    return overlap >= settings.gate2_presence_overlap_fraction, overlap

# Verdicts that block grounding. "external_unsupported" is reported as its own
# label for transparency but counts against grounding exactly like
# "not_supported", so it triggers regeneration and eventual abstention.
_BLOCKING_VERDICTS = {
    "not_supported",
    "external_unsupported",
    "caption_only_uncertified",
    "false_hedge",
}


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
    # Set for caption_only_uncertified: where the reader should look instead
    # of being handed the claim as an asserted fact.
    figure_reference: str | None = None
    # Set for false_hedge: the denied topic that evidence actually supports, so
    # the regeneration prompt can tell the generator what to re-address.
    probe_topic: str | None = None


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


@traceable(name="check_groundedness")
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
            # An absence claim ("the documents do not discuss X") has no
            # positive assertion to entail, so instead of checking the claim we
            # check its DENIAL: is the topic X actually PRESENT in the evidence?
            # The right question is lexical ("do the topic's distinctive terms
            # occur in the retrieved text") rather than entailment -- NLI reads
            # a meta-probe like "the text discusses X" as neutral even when X is
            # plainly there (measured 0.36 on a topic the corpus states in
            # full). So we reuse the SAME token-overlap presence check used for
            # unscorable claims, run over ALL retrieved hits (an absence claim
            # cites nothing, so there is no cited-chunk subset to restrict to).
            # If the topic is present, the hedge is false -> "false_hedge",
            # which blocks grounding and feeds the existing regeneration loop.
            #
            # With no extractable topic there is nothing to probe, so the claim
            # falls back to the old unverified pass-through (non-blocking).
            topic = (claim.topic or "").strip()
            if not topic or not answer.hits_used:
                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim.text,
                        claim_type=claim.claim_type,
                        verdict="absence_unverified",
                    )
                )
                continue

            present, overlap = _presence_supported(
                topic, [hit.raw_text for hit in answer.hits_used]
            )
            if present:
                logger.info(
                    "False hedge: claim denies a topic the evidence contains "
                    "(token overlap=%.2f): %r", overlap, topic[:100]
                )
                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim.text,
                        claim_type=claim.claim_type,
                        verdict="false_hedge",
                        entailment_prob=overlap,
                        probe_topic=topic,
                    )
                )
            else:
                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim.text,
                        claim_type=claim.claim_type,
                        verdict="absence_unverified",
                        entailment_prob=overlap,
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

        if claim.claim_type == "unscorable":
            # Formula / figure-reference claims: verify the artifact really
            # occurs in the cited evidence instead of asking NLI to entail it.
            present, overlap = _presence_supported(
                claim.text, [premise for _, premise in resolved]
            )
            verdicts.append(
                ClaimVerdict(
                    claim_text=claim.text,
                    claim_type=claim.claim_type,
                    verdict="supported_presence" if present else "not_supported",
                    evidence_chunk_ids=all_chunk_ids_for(resolved),
                    entailment_prob=overlap,
                )
            )
            continue

        threshold = settings.gate2_entailment_threshold

        # (b) score the claim against every contributing chunk in ONE batched
        # call, each chunk scored on its best sentence window. Scoring all of
        # them (rather than breaking at the first success) also gives the
        # external corroboration rule below the complete set of supporting
        # sources, which an early break used to hide.
        results = score_claim_against_premises(
            claim.text, [premise for _, premise in resolved]
        )
        best_prob = max(
            (result.entailment_prob for result in results), default=None
        )
        supporting = [
            (chunk_id, result.entailment_prob)
            for (chunk_id, _), result in zip(resolved, results)
            if result.entailment_prob >= threshold
        ]

        all_chunk_ids = [chunk_id for chunk_id, _ in resolved]

        if supporting:
            # PROVENANCE TIER: a claim may only be certified if its SUPPORT
            # SET includes at least one extracted_text chunk. Support resting
            # solely on a vision caption is not asserted -- the reader is
            # pointed at the figure instead. Without this, a bad caption could
            # silently certify a claim as grounded.
            if all(
                _is_model_generated(chunk_id, hit_by_chunk_id)
                for chunk_id, _ in supporting
            ):
                caption_ids = [chunk_id for chunk_id, _ in supporting]
                logger.info(
                    "Claim supported only by model-generated caption(s); not "
                    "certifying: %r",
                    claim.text[:100],
                )
                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim.text,
                        claim_type=claim.claim_type,
                        verdict="caption_only_uncertified",
                        evidence_chunk_ids=caption_ids,
                        entailment_prob=max(prob for _, prob in supporting),
                        figure_reference=_figure_pointer(
                            caption_ids, hit_by_chunk_id
                        ),
                    )
                )
                continue

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
            combined_result = check_entailment_windowed(combined, claim.text)
            if best_prob is None or combined_result.entailment_prob > best_prob:
                best_prob = combined_result.entailment_prob
            if combined_result.entailment_prob >= threshold:
                # PROVENANCE TIER, combination path. The single-chunk path above
                # refuses to certify on model-generated (caption) support alone;
                # this path must too, or a claim entailed only by captions that
                # jointly (but not individually) entail it would be certified.
                # There is no per-chunk support set here -- that is why we are in
                # the combination path -- so the rule is: the claim must still be
                # entailed by the certifiable chunks ALONE. If a caption is
                # needed to reach the threshold, the claim is not certified.
                # Reaching this branch means every chunk individually scored
                # below threshold, so fewer than two certifiable chunks cannot
                # entail it jointly and no extra NLI call is needed.
                generated = [
                    chunk_id
                    for chunk_id, _ in resolved
                    if _is_model_generated(chunk_id, hit_by_chunk_id)
                ]
                support_ids = all_chunk_ids
                if generated:
                    certifiable = [
                        (chunk_id, text)
                        for chunk_id, text in resolved
                        if chunk_id not in generated
                    ]
                    certified = False
                    if len(certifiable) > 1:
                        certified_result = check_entailment_windowed(
                            "\n\n".join(text for _, text in certifiable)[
                                : settings.gate2_combination_max_chars
                            ],
                            claim.text,
                        )
                        certified = certified_result.entailment_prob >= threshold
                    if not certified:
                        logger.info(
                            "Claim jointly supported only with model-generated "
                            "caption(s); not certifying: %r",
                            claim.text[:100],
                        )
                        verdicts.append(
                            ClaimVerdict(
                                claim_text=claim.text,
                                claim_type=claim.claim_type,
                                verdict="caption_only_uncertified",
                                evidence_chunk_ids=generated,
                                entailment_prob=combined_result.entailment_prob,
                                figure_reference=_figure_pointer(
                                    generated, hit_by_chunk_id
                                ),
                            )
                        )
                        continue
                    # Certified without the captions: they are not part of the
                    # support set that decided the verdict.
                    support_ids = [chunk_id for chunk_id, _ in certifiable]
                has_local = any(
                    not _is_external(chunk_id, hit_by_chunk_id)
                    for chunk_id in support_ids
                )
                urls = _distinct_external_urls(support_ids, hit_by_chunk_id)
                verdicts.append(
                    ClaimVerdict(
                        claim_text=claim.text,
                        claim_type=claim.claim_type,
                        verdict=(
                            "supported_combination"
                            if (has_local or len(urls) >= 2)
                            else "external_unsupported"
                        ),
                        evidence_chunk_ids=support_ids,
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

    # not_supported, external_unsupported, caption_only_uncertified and
    # false_hedge all block grounding, so each feeds the corrective loop and
    # the abstention message. absence_unverified (a hedge we could not probe)
    # and meta_excluded deliberately do NOT count against grounding.
    unsupported = [v for v in verdicts if v.verdict in _BLOCKING_VERDICTS]
    return Gate2Verdict(
        overall_grounded=not unsupported,
        claims=verdicts,
        unsupported_claims=unsupported,
    )
