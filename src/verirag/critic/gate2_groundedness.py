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
from verirag.critic.claims import _QUOTED_RE, Claim, extract_claims
from verirag.critic.nli import (
    check_entailment_windowed,
    normalize_premise,
    score_claim_against_premises,
)
from verirag.critic.table_claims import verify_numeric_claim
from verirag.critic.visual_check import VisualChecker
from verirag.generation.generator import GeneratedAnswer
from verirag.retrieval.types import SearchHit

logger = logging.getLogger(__name__)

ClaimVerdictLabel = Literal[
    "supported_single",
    "supported_combination",
    "supported_presence",
    "caption_only_uncertified",
    "visually_verified",
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

# Words that name the structure rather than say anything about it, so a
# structural claim's CONTENT check ignores them ("the section ... covers X").
_STRUCTURAL_STOPWORDS = _PRESENCE_STOPWORDS | frozenset(
    """sections chapter chapters heading headings headed titled title titles
    header headers subsection related covers cover discusses discuss passages
    mentioned mention contains contain named called appears appear notes""".split()
)
_SQUASH_RE = re.compile(r"\W+")


def _squash(text: str) -> str:
    """Lower-case and collapse everything but word characters, so a title that
    a PDF broke across lines, or typeset with different punctuation, still
    matches verbatim."""
    return _SQUASH_RE.sub(" ", text.lower()).strip()


def _stem(token: str) -> str:
    """Crude prefix stem so morphological variants match ("relation" /
    "relationship", "German" / "Germans"). Used only where a miss would
    reject a true claim for want of an inflection."""
    return token[:6]


def _content_overlap(text: str, premises: list[str], stopwords=_STRUCTURAL_STOPWORDS) -> float | None:
    """Share of ``text``'s distinctive tokens found in ``premises``; ``None`` if it has none.

    Compared on prefix stems (see :func:`_stem`).
    """
    tokens = {_stem(t.lower()) for t in _TOKEN_RE.findall(text) if t.lower() not in stopwords}
    if not tokens:
        return None
    evidence = {
        _stem(t) for t in _TOKEN_RE.findall(normalize_premise(" ".join(premises)).lower())
    }
    return len(tokens & evidence) / len(tokens)


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers_in_evidence(text: str, premises: list[str]) -> bool | None:
    """True if every number in ``text`` also occurs in ``premises``; ``None`` if it has none.

    A claim that fails entailment although all of its numbers are in the cited
    text was most likely paraphrased past what NLI can follow (typical of a value
    read off a markdown table); one that states a number the evidence lacks is
    the kind to drop.
    """
    wanted = {n.replace(",", "") for n in _NUMBER_RE.findall(text)}
    if not wanted:
        return None
    have = {n.replace(",", "") for n in _NUMBER_RE.findall(" ".join(premises))}
    return wanted <= have


def _check_structural(
    claim: Claim,
    resolved: list[tuple[str, str]],
    hit_by_chunk_id: dict[str, SearchHit],
) -> "ClaimVerdict":
    """Verify a structural claim by PRESENCE rather than entailment.

    "Passage 1 contains a section heading titled X" asks whether X literally
    appears in the cited text -- a substring question. An NLI model asks
    whether a sentence pair is semantically consistent and scored such claims
    0.0035-0.22 on headings that were there. A claim of the form "the section
    titled X covers Y" is checked in two parts: every quoted title must appear
    verbatim (modulo case and punctuation) in the cited evidence, and Y's
    distinctive words must occur there too, at the same overlap fraction used
    for the other presence checks.

    The provenance rule still applies: a title that appears only in a
    model-written figure caption is not certified.
    """
    chunk_ids = [chunk_id for chunk_id, _ in resolved]
    premises = [text for _, text in resolved]
    squashed = {chunk_id: _squash(text) for chunk_id, text in resolved}

    found_ids: list[str] = []
    for title in claim.titles:
        key = _squash(title)
        ids = [cid for cid, text in squashed.items() if key and key in text]
        if not ids:
            return ClaimVerdict(
                claim_text=claim.text,
                claim_type=claim.claim_type,
                verdict="not_supported",
                evidence_chunk_ids=chunk_ids,
                entailment_prob=0.0,
                lexical_overlap=0.0,
            )
        found_ids.extend(i for i in ids if i not in found_ids)

    if found_ids and all(_is_model_generated(i, hit_by_chunk_id) for i in found_ids):
        return ClaimVerdict(
            claim_text=claim.text,
            claim_type=claim.claim_type,
            verdict="caption_only_uncertified",
            evidence_chunk_ids=found_ids,
            entailment_prob=1.0,
            figure_reference=_figure_pointer(found_ids, hit_by_chunk_id),
        )

    overlap = _content_overlap(_QUOTED_RE.sub(" ", claim.text), premises)
    if overlap is None and not claim.titles:
        return ClaimVerdict(
            claim_text=claim.text,
            claim_type=claim.claim_type,
            verdict="not_supported",
            evidence_chunk_ids=chunk_ids,
            entailment_prob=0.0,
        )
    supported = overlap is None or overlap >= settings.gate2_presence_overlap_fraction
    return ClaimVerdict(
        claim_text=claim.text,
        claim_type=claim.claim_type,
        verdict="supported_presence" if supported else "not_supported",
        evidence_chunk_ids=found_ids or chunk_ids,
        entailment_prob=1.0 if overlap is None else overlap,
        lexical_overlap=overlap,
    )


def _verified_by_table(
    claim_text: str,
    resolved: list[tuple[str, str]],
    hit_by_chunk_id: dict[str, SearchHit],
) -> str | None:
    """Chunk id of a cited TABLE chunk that backs ``claim_text`` cell-for-cell, else None.

    Only table chunks are consulted, so this never changes how a claim resting on
    text or caption evidence is checked. A claim it cannot verify is not
    penalised here -- it simply falls through to NLI as before.
    """
    for chunk_id, text in resolved:
        hit = hit_by_chunk_id.get(chunk_id)
        if hit is None or hit.metadata.get("content_type") != "table":
            continue
        if verify_numeric_claim(claim_text, text):
            return chunk_id
    return None


def _caption_claim_verdict(
    claim,
    caption_ids: list[str],
    hit_by_chunk_id: dict[str, SearchHit],
    entailment_prob: float | None,
    checker: "VisualChecker | None",
) -> "ClaimVerdict":
    """``visually_verified`` if the FIGURE shows the claim, else the old rejection.

    The caption still certifies nothing: the claim is put to the vision model
    against the figure's own pixels, and only an unambiguous YES certifies it. NO,
    CANNOT_TELL, a missing image or an exhausted budget all keep
    ``caption_only_uncertified``.
    """
    if checker is not None:
        for chunk_id in caption_ids:
            hit = hit_by_chunk_id.get(chunk_id)
            if hit is None:
                continue
            if checker.check(claim.text, hit) == "YES":
                logger.info("Claim certified by the figure itself: %r", claim.text[:100])
                return ClaimVerdict(
                    claim_text=claim.text,
                    claim_type=claim.claim_type,
                    verdict="visually_verified",
                    evidence_chunk_ids=[chunk_id],
                    entailment_prob=entailment_prob,
                )
    return ClaimVerdict(
        claim_text=claim.text,
        claim_type=claim.claim_type,
        verdict="caption_only_uncertified",
        evidence_chunk_ids=caption_ids,
        entailment_prob=entailment_prob,
        figure_reference=_figure_pointer(caption_ids, hit_by_chunk_id),
    )


def _best_hits_for_topic(
    topic: str, hits: list[SearchHit], limit: int = 2
) -> list[tuple[str, float]]:
    """``(chunk_id, overlap)`` for the hits sharing most of ``topic``'s words, best first.

    Each hit is scored ALONE. Pooling words across every retrieved chunk (as the
    first version did) made a long topic look "present" whenever its words were
    scattered over the context, e.g. a true "no other chapter names address X"
    scored 0.71 pooled but 0.57 in its best single chunk.
    """
    scored = [
        (hit.chunk_id, _presence_supported(topic, [hit.raw_text])[1]) for hit in hits
    ]
    scored.sort(key=lambda pair: -pair[1])
    return [pair for pair in scored[:limit] if pair[1] > 0]


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
    # For not_supported claims: the share of the claim's distinctive words that
    # DO occur in its cited evidence. High overlap with a failed entailment
    # means the claim was probably paraphrased past what NLI can follow, not
    # invented -- the regeneration prompt treats the two differently.
    lexical_overlap: float | None = None
    # For not_supported claims that state numbers: whether every one of them
    # occurs in the cited evidence (None if the claim has no numbers).
    numbers_in_evidence: bool | None = None


@dataclass
class Gate2Verdict:
    """Groundedness outcome for a whole answer."""

    overall_grounded: bool
    claims: list[ClaimVerdict]
    unsupported_claims: list[ClaimVerdict]
    # Every visual re-check made for this answer (claim, image, verdict).
    visual_checks: list = field(default_factory=list)


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
    # Per-answer budget for visual re-checks of caption-only claims.
    checker = VisualChecker()
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
            # unscorable claims, run over EACH retrieved hit separately and taking
            # the best (an absence claim cites nothing, so there is no cited-chunk
            # subset to restrict to). The words must co-occur in ONE chunk: pooled
            # across all hits, almost any long topic looks present.
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

            best_hits = _best_hits_for_topic(topic, answer.hits_used)
            overlap = best_hits[0][1] if best_hits else 0.0
            present = overlap >= settings.gate2_presence_overlap_fraction
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
                        evidence_chunk_ids=[chunk_id for chunk_id, _ in best_hits],
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

        if claim.claim_type == "structural":
            verdicts.append(_check_structural(claim, resolved, hit_by_chunk_id))
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

        # A numeric claim whose every number is paired with a correctly named
        # row and column of a cited TABLE chunk is verified by lookup: NLI reads
        # a markdown table poorly (0.0057-0.49 on true cell values). Table chunks
        # are document-derived, so this is certifying evidence like any text.
        table_chunk = _verified_by_table(claim.text, resolved, hit_by_chunk_id)
        if table_chunk is not None:
            verdicts.append(
                ClaimVerdict(
                    claim_text=claim.text,
                    claim_type=claim.claim_type,
                    verdict="supported_presence",
                    evidence_chunk_ids=[table_chunk],
                    entailment_prob=1.0,
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
                verdicts.append(_caption_claim_verdict(
                    claim, caption_ids, hit_by_chunk_id,
                    max(prob for _, prob in supporting), checker,
                ))
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
                        verdicts.append(_caption_claim_verdict(
                            claim, generated, hit_by_chunk_id,
                            combined_result.entailment_prob, checker,
                        ))
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
                lexical_overlap=_content_overlap(
                    claim.text, [text for _, text in resolved], _PRESENCE_STOPWORDS
                ),
                numbers_in_evidence=_numbers_in_evidence(
                    claim.text, [text for _, text in resolved]
                ),
            )
        )

    # not_supported, external_unsupported, caption_only_uncertified and
    # false_hedge all block grounding, so each feeds the corrective loop and
    # the abstention message. absence_unverified (a hedge we could not probe)
    # and meta_excluded deliberately do NOT count against grounding.
    unsupported = [v for v in verdicts if v.verdict in _BLOCKING_VERDICTS]
    if checker.checks:
        logger.info("Gate 2 made %d visual check(s) (%d billed): %s",
                    len(checker.checks), checker.used,
                    [(c.verdict, c.image[-40:]) for c in checker.checks])
    return Gate2Verdict(
        overall_grounded=not unsupported,
        claims=verdicts,
        unsupported_claims=unsupported,
        visual_checks=[vars(c) for c in checker.checks],
    )
