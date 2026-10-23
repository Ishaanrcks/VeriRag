# QueensLand targeted re-run — 2026-10-23 (post critic calibration)

Baseline: `queensland_corpus_eval_2026-10-23.md` (pre-fix, same 21 questions).
Store: `eval_store/queensland/`, deleted and re-ingested from scratch in a
single process after a concurrent-write incident made the previous state
untrustworthy. Mode: `corpus_only=True`, `AnswerFormat.CONCISE`.
Raw traces: `queensland_subset_eval_2026-10-23.json`.

Fixes under test: `2bfdeed` (windowed premises, unscorable claims),
`d770745` (Gate 1 re-tune), `bdfd819` (NLI batch bucketing, strided windows),
`a5d48b3` + `70f9591` (diversity top-k, duplicate citation merging).

## Corpus (re-ingested clean)

400 chunks across 6 documents, Chroma and BM25 in sync (400 = 400), 11 min.
`GFM RAG.pdf` 66 · `2502.01113v3.pdf` 85 · `2605.04847v2.pdf` 89 ·
`EFFICIENT PROBABILISTIC LOGIC…` 47 · `G-Reasoner.pdf` 71 ·
`OnSelfDistillation.pdf` 42. The duplicate pair shares group `dc94fd1ff22c`;
the other four groups are distinct.

## Subset rationale

Running "every false negative + every correct abstention + 3 easy ones" would
be 20 of 21 questions, so it was prioritised to 14:

* **A — false negatives to convert (7):** the six named (Q2, Q4, Q6, Q11, Q13,
  Q18) plus Q8, the only question exercising the formula/unscorable path.
* **B — correct refusals that must hold (4):** Q16, Q19, Q20, Q21. A
  regression here matters more than any recall gain.
* **C — previously-correct answers (3):** Q1, Q3, Q9, one per distinct paper.

Skipped: Q5, Q7, Q10, Q14, Q15, Q17 (more Gate 2 false negatives of the same
kind as Q2/Q6/Q11) and Q12 (a second G-Reasoner easy case).

## Before / after

| Q | grp | before | after | required | result |
|---|---|---|---|---|---|
| 1 | C | answered | answered | answer | ✅ |
| 2 | A | abstained | **answered** | answer | ✅ converted |
| 3 | C | answered | answered | answer | ✅ |
| 4 | A | abstained | **answered** | answer | ✅ converted |
| 6 | A | abstained | abstained | answer | ❌ still failing |
| 8 | A | abstained | **answered** | answer | ✅ converted |
| 9 | C | answered | answered | answer | ✅ |
| 11 | A | abstained | **answered** | answer | ✅ converted |
| 13 | A | partial | refused | answer | ❌ still failing |
| 16 | B | abstained | abstained | decline | ✅ held |
| 18 | A | partial | abstained | answer | ❌ still failing |
| 19 | B | abstained | abstained | decline | ✅ held |
| 20 | B | abstained | refused | decline | ✅ held |
| 21 | B | refused | refused | decline | ✅ held |

```
A. false negatives converted : 4/7   (still failing: 6, 13, 18)
B. correct refusals held     : 4/4
C. easy cases still correct  : 3/3
avg regenerations            : 0.86   (was 1.43 across the full 21)
```

## The result that matters most

**No precision regression. All 4 previously-correct refusals still decline.**
Nothing was traded away: no question that should refuse now answers.

Group B behaviour in detail — Q16 and Q19 abstain (corpus-level aggregation
the system does not do), Q20 and Q21 decline explicitly via the generation
prompt rather than burning regenerations. Q20 improved from "abstained after 2
regenerations" to "refused after 1", and Q21 from 1 regeneration to 0. The
refusals got *cheaper*, not weaker.

## Converted answers verified against the papers

"Answered" is not "correct", so each was checked against the source:

* **Q2** — "60 knowledge graphs containing over 14M triples and 700k
  documents" ✅ exact.
* **Q4** — "a four-layer abstraction consisting of an attribute layer, a
  knowledge graph layer, a document layer, and a community layer" ✅. This is
  the Bug 2 case whose pre-fix Gate 1 refusal *described the four layers and
  returned INSUFFICIENT anyway*.
* **Q8** — "dual-head architecture that decouples the estimation of the
  prediction ŷ and the interval width d̂" ✅. The formula claim
  `H = GNN(X, E)` that previously sank this answer is now classified
  `unscorable` and checked by token presence.
* **Q11** — "OPSD's generation is capped at 1024 tokens, while GRPO's is
  capped at 16k" ✅ exact.

Zero wrong answers among the four conversions, so the recall gain did not come
at the cost of correctness.

## Still failing (3 of 7)

* **Q6** (ExpressGNN = MLN + GNN) — Gate 2, abstained after 2 regenerations.
  The windowed-premise battery scores this claim 0.69, above threshold, so the
  failure is now in the *generated phrasing* of the claims rather than the
  premise handling.
* **Q13** (G-reasoner vs GFM-RAG params + limitation) — declined, but the
  answer correctly states G-reasoner's side: "GFM-RAG is specifically designed
  for knowledge graphs and cannot be directly applied to other types of
  graph-structured knowledge". It refused only on the parameter comparison,
  because GFM-RAG's 8M figure was not among the retrieved chunks. Eight final
  chunks still cannot hold both halves of a cross-paper question.
* **Q18** (shared authors) — abstained. Author lists are front-matter that
  retrieval does not favour for a semantic query.

All three are coverage/phrasing limits, not grounding failures: none produced
a wrong answer.

## New bug found during this run

**`_ROW_METADATA_KEYS` in `sparse_index.py` omits `duplicate_group_id` and
`parent_text`** — the third instance of this same pattern in that one tuple
(`source_type` was the first, fixed in `775134c`). Measured:

```
SPARSE: dup_group=None           parent_text=NO
DENSE:  dup_group='dc94fd1ff22c' parent_text=yes
```

Two consequences:

1. **Bug 5's citation merging silently fails on the sparse path.** Visible in
   this run: Q2 cited `GFM RAG.pdf` and `2502.01113v3.pdf` as two separate
   citations, and Q4 listed three separate `2502.01113v3.pdf` citations, even
   though all share a duplicate group.
2. **Small-to-big degrades for sparse-only hits.** With no `parent_text`,
   `build_context_block` falls back to the small chunk, so generation loses
   the parent context Phase 5 added — silently, on one retrieval path only.

Impact on *this* eval was cosmetic: all four converted answers were correct.
But the hand-maintained allowlist has now failed three times, so the fix
should be structural rather than another key: hydrate sparse-only hits'
metadata from Chroma by `chunk_id` in one batched `get` at retrieval time,
which keeps the JSONL small and cannot miss a future field.

## Other observations

* **Gate 2 remains the cost centre.** Each claim scores against ~8 chunks and
  a 512-token forward pass costs ~700 ms, so a question runs minutes. The
  14-question subset took roughly 35 min after an 11-min ingest.
* **Figure captioning keeps losing figures.** QpiGNN alone dropped ~25 to
  empty vision responses; it ingested 89 text chunks but only 6 figure chunks.
* **Early dedup now works.** `ingest_file` checks `doc_id` before loading, so
  the subset run's ingest pass skipped all 6 papers instantly instead of
  re-running figure captioning on each.
