"""Tiered document selection for MMLongBench-Doc ingestion.

The benchmark's 200-question subset references 103 PDFs. Ingesting all of them is
slow and memory-hungry on an 8 GB machine, and the point of tiering is to measure
how retrieval and the gates behave as the corpus GROWS (10 -> 30 -> 60 -> 103
documents) before paying for everything. So the tiers must NEST (tier 2 extends
tier 1) and each must be a fair sample of the question mix, not 10 documents that
happen to be all text.

``rank_documents`` orders every candidate by a greedy coverage rule:

  1. most NEW labels covered, where a document's labels are the question groups it
     carries (text / table / chart / figure / layout / multiple / unanswerable),
     ``cross_page`` if any of its questions needs several pages, and its document
     type (slides, papers, reports... ingest very differently);
  2. then the document with more questions (a document nobody asks about teaches
     us nothing);
  3. then the smaller file (cheaper to ingest);
  4. then the name, so the ranking is deterministic.

Documents over ``max_bytes`` are excluded: file size is the only cost known before
downloading, and one 30 MB brochure would cost more than five ordinary documents.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

DEFAULT_MAX_BYTES = 4_000_000


@dataclass
class DocProfile:
    doc_id: str
    questions: int = 0
    doc_type: str = ""
    groups: collections.Counter = field(default_factory=collections.Counter)
    cross_page: bool = False

    @property
    def labels(self) -> frozenset[str]:
        found = {f"group:{g}" for g in self.groups}
        if self.cross_page:
            found.add("cross_page")
        if self.doc_type:
            found.add(f"doc_type:{self.doc_type}")
        return frozenset(found)


def doc_profiles(questions: list[dict]) -> dict[str, DocProfile]:
    """One profile per document, from the question set."""
    profiles: dict[str, DocProfile] = {}
    for q in questions:
        p = profiles.setdefault(q["doc_id"], DocProfile(q["doc_id"], doc_type=q.get("doc_type") or ""))
        p.questions += 1
        p.groups[q["group"]] += 1
        p.cross_page = p.cross_page or bool(q.get("cross_page"))
    return profiles


def rank_documents(
    questions: list[dict],
    sizes: dict[str, int],
    *,
    already: list[str] | tuple[str, ...] = (),
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[str]:
    """Candidate documents, best first, never including ``already`` (the earlier tier).

    Coverage is measured against the labels ``already`` covers, so extending a tier
    adds what it is missing rather than repeating it.
    """
    profiles = doc_profiles(questions)
    covered: set[str] = set()
    for doc in already:
        if doc in profiles:
            covered |= profiles[doc].labels
    pool = [d for d, p in profiles.items()
            if d not in already and d in sizes and sizes[d] <= max_bytes]

    ranked: list[str] = []
    while pool:
        best = min(
            pool,
            key=lambda d: (-len(profiles[d].labels - covered), -profiles[d].questions,
                           sizes[d], d),
        )
        ranked.append(best)
        covered |= profiles[best].labels
        pool.remove(best)
    return ranked


def coverage(questions: list[dict], docs: list[str]) -> dict:
    """What a set of documents covers, for the manifest and the report."""
    profiles = doc_profiles(questions)
    chosen = [profiles[d] for d in docs if d in profiles]
    groups: collections.Counter = collections.Counter()
    for p in chosen:
        groups.update(p.groups)
    return dict(
        documents=len(chosen),
        questions=sum(p.questions for p in chosen),
        groups=dict(sorted(groups.items())),
        doc_types=sorted({p.doc_type for p in chosen if p.doc_type}),
        cross_page_documents=sum(p.cross_page for p in chosen),
        all_groups_present=sorted({g for g in (q["group"] for q in questions)} - set(groups)) == [],
    )
