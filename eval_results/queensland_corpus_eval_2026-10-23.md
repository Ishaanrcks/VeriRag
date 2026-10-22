# QueensLand corpus evaluation — 2026-10-23

Corpus: `/Users/ishanyashpatel/Documents/QueensLand` (6 PDFs)
Store: isolated at `./eval_store/queensland/` (not the live Streamlit store)
Mode: `corpus_only=True` for every question, `AnswerFormat.CONCISE`
Raw traces: `queensland_traces_2026-10-23.json`

## Corpus

| Document | Pages | Text chunks | Figure chunks |
|---|---|---|---|
| `GFM RAG.pdf` | 19 | 54 | 11 |
| `Papers/2502.01113v3.pdf` | 35 | 75 | 11 |
| `Papers/2605.04847v2.pdf` | 28 | 83 | 6 |
| `Papers/EFFICIENT PROBABILISTIC LOGIC REASONING WITH GRAPH NEURAL NETWORKS.pdf` | 20 | 46 | 1 |
| `Papers/G-Reasoner.pdf` | 25 | 61 | 11 |
| `Papers/OnSelfDistillation.pdf` | 15 | 42 | 0 |
| **Total** | **142** | **361** | **40** |

Two corpus facts established by reading, not inference:

1. **`GFM RAG.pdf` and `2502.01113v3.pdf` are the same paper** — GFM-RAG (Luo, Zhao
   et al.), 19pp camera-ready vs 35pp arXiv. Different content hashes, so they
   ingest as two distinct `doc_id`s.
2. **G-Reasoner is the documented successor to GFM-RAG** — shared first authors and
   project page, and G-Reasoner explicitly cites "GFM-RAG (Luo et al., 2025) …
   8M parameters … still limited to specific knowledge graphs and cannot
   generalize to other graph structures." Synthesis questions rest on this stated
   link rather than an assumed one.

`QpiGNN` (quantile-free UQ for GNN node regression), `ExpressGNN` (MLN + GNN
variational inference) and `OPSD` (on-policy LLM self-distillation, no graph
component) are genuinely unrelated to each other in subject, which is what makes
the false-premise questions false rather than merely hard.

## Results

Expected labels were written before running. "Correct?" scores **substance**, not
the mechanical verdict: the system can decline in two ways — abstain via a gate,
or answer with an explicit "the documents do not address this", which Fix 2's
generation prompt now encourages. Both count as a correct decline.

| # | Question | Expected | Verdict | Correct? | Notes |
|---|---|---|---|---|---|
| 1 | GFM-RAG parameter count | answerable | answered | ✅ | "8M parameters" — exact. Cited **both** duplicate copies, consuming 5 citation slots for one paper. |
| 2 | GFM-RAG training scale (60 KGs / 14M triples / 700k docs) | answerable | abstained | ❌ FN | Gate 2. The rejected claim text itself begins "two-stage training used large-scale datasets comprising 6…" — the right answer was generated then discarded. |
| 3 | G-reasoner parameter count | answerable | answered | ✅ | "34M parameters" — exact, 0 regenerations. |
| 4 | What is QuadGraph, how many layers | answerable | abstained | ❌ FN | **Gate 1**, and the judge's own refusal states the answer: "describe it as a 4-layer graph structure (attribute, knowledge graph, document, community layers), but they do not expl…". See Bug 2. |
| 5 | ExpressGNN's four benchmark datasets | answerable | abstained | ❌ FN | Gate 2; claim was "evaluated on four benchmark dat…". |
| 6 | What ExpressGNN combines (MLN + GNN) | answerable | abstained | ❌ FN | Gate 2; claim was correct ("combines Markov Logic Networks (MLNs) and graph neural n…"). |
| 7 | QpiGNN coverage / interval-width gains | answerable | abstained | ❌ FN | Gate 2; claim referenced the 19 benchmarks correctly. |
| 8 | QpiGNN dual-head architecture | answerable | abstained | ❌ FN | Gate 2. One rejected claim was a formula, "a GNN encoder computes node embeddings H = GNN(X, E)" — not an NLI-checkable proposition. See Bug 3. |
| 9 | Qwen3 sizes in OPSD | answerable | answered | ✅ | "Qwen3-1.7B, Qwen3-4B, and Qwen3-8B" — exact. |
| 10 | OPSD teacher's privileged information | answerable | abstained | ❌ FN | Gate 2; claim was correct. |
| 11 | OPSD vs GRPO token caps (1024 / 16k) | answerable | abstained | ❌ FN | Gate 2; claim cited Figure 3's caps correctly. |
| 12 | G-reasoner benchmark count | answerable | answered | ✅ | "six benchmark datasets", plus correct names (HotpotQA, MuSiQue, 2WikiMultiHopQA, G-bench Novel/Medical/CS). |
| 13 | G-reasoner vs GFM-RAG params + stated limitation | answerable | answered | ⚠️ partial | Honest partial refusal: correctly reported G-reasoner's side but said the sources "do not state GFM-RAG's parameter count" — true of the 5 retrieved chunks, false of the corpus (Q1 found it). Retrieval gap, not a hallucination. |
| 14 | GFM-RAG vs G-reasoner graph representation | answerable | abstained | ❌ FN | Gate 2. |
| 15 | ExpressGNN vs GFM-RAG generalization claims | answerable | abstained | ❌ FN | Gate 2. |
| 16 | Which papers use GNNs, for what | *reclassified* unanswerable | abstained | ✅ | **Correct abstention.** Needs corpus-level aggregation; the model's own claim ("two papers") was also wrong (truth: four), and the aggregate claims carried no citations so Gate 2 found "no resolvable evidence chunks". My original "answerable" label was wrong. |
| 17 | How each scales training | answerable | abstained | ❌ FN | Gate 2; the model also mischaracterized GFM-RAG as "only a baseline", so abstention was defensible here. |
| 18 | Do GFM-RAG and G-reasoner share authors | answerable | answered | ⚠️ partial | Listed G-reasoner's 12 authors correctly, then declined to confirm overlap because GFM-RAG's author list wasn't retrieved. Retrieval gap. |
| 19 | Do two documents describe the same paper | unanswerable | abstained | ✅ | Correct — needs corpus-level aggregation the system does not perform. |
| 20 | UQ method "all these papers" use | false premise | abstained | ✅ | Correctly refused the presupposition. |
| 21 | How OPSD improves GFM-RAG retrieval | false premise | answered | ✅ | **Correct decline in substance**: "The provided documents do not address this question… neither passage discusses any interaction between the two." Fix 2's context-adherence prompt working as designed. |

## Summary statistics

| Metric | Value |
|---|---|
| Questions | 21 |
| Mechanical answer rate | 7 / 21 (33%) |
| Mechanical abstention rate | 14 / 21 (67%) |
| **Substantively correct behavior** | **10 / 21 (48%)** |
| Fully correct answers | 4 (Q1, Q3, Q9, Q12) |
| Correct declines (false premise / unanswerable) | 4 (Q16, Q19, Q20, Q21) |
| Partial, honest-but-incomplete answers | 2 (Q13, Q18) |
| **Accuracy among substantive answers** | **4 / 4 = 100%** |
| **Critic false positives (wrong answer approved)** | **0** |
| Critic false negatives (answerable → abstained) | 11 |
| Avg regeneration count | 1.43 |
| Loop trigger rate | 0.81 |
| Avg estimated cost / query | $0.0094 |

p95 latency by node (ms): `gate2` **13,095**, `retrieve` 2,756, `generate` 1,936,
`regenerate` 1,907, `resolve_intent` 102, `gate1` 3.7.

**The headline:** precision is perfect and recall is the problem. The system never
produced a wrong or hallucinated answer, never approved an ungrounded claim, and
got all four decline cases right. But it abstained on 11 of 17 genuinely
answerable questions, and in most of those the abstention message *contains the
correct answer* — it was generated, then rejected.

## Bugs to triage

### Bug 1 (highest value) — Gate 2's all-or-nothing aggregation
`overall_grounded` is `False` if **any single** claim fails entailment. Claim
extraction decomposes a ~200–700 char answer into 15–25 atomic claims, so a
per-claim false-negative rate of only 10–15% makes whole-answer rejection close
to certain. This single rule accounts for 10 of the 11 false negatives.

Evidence it is a scoring artifact and not bad generation: Q2, Q5, Q6, Q7, Q10 and
Q11 all show the *correct* fact inside the rejected-claim list.

Candidate directions: require a supported *fraction* rather than unanimity; block
only on claims NLI actively **contradicts** (rather than merely finds neutral —
Phase 7 already showed neutral means "can't tell"); or weight claims by whether
they're load-bearing for the question.

### Bug 2 — Gate 1 over-strictness, likely introduced by Fix 2
Q4's judge wrote: *"The passages define QuadGraph and describe it as a 4-layer
graph structure (attribute, knowledge graph, document, community layers), but they
do not expl…"* and returned INSUFFICIENT. It had the answer and refused anyway.
Fix 2's added specificity test and the instruction to answer INSUFFICIENT
"whenever a correct answer would require knowledge you have but the passages do
not contain" plausibly over-corrected. This is a regression I introduced and
should be re-tuned — carefully, since the same tightening is what makes Q20/Q21
behave correctly.

### Bug 3 — Claim extraction emits structurally unverifiable claims
Observed claim types NLI cannot score meaningfully:
- **Formulas**: `"a GNN encoder computes node embeddings H = GNN(X, E)"` (Q8)
- **Figure/table references**: `"The paper's Figure 3 description states…"` (Q11)
- **Attributive framings**: `"The paper states that…"` (pervasive; Phase 7 measured
  these as highly phrasing-sensitive)
- **Uncited corpus-level aggregates**: `"Two papers in the corpus apply graph
  neural networks"` → logged "no resolvable evidence chunks" (Q16)

These should be classified out of scoring (like `meta`) or rewritten into direct
propositional form at extraction time.

### Bug 4 — `retrieval_final_top_k=5` is too small for synthesis
Q13 and Q18 failed because five chunks drawn from a 401-chunk, 6-paper corpus
cannot hold facts from two different papers at once. Both produced honest partial
refusals rather than errors, but both are answerable. A larger final-k, or
per-document quota, would fix them.

### Bug 5 — Duplicate documents crowd out citation slots
Q1 returned 5 citations, all of them the *same* paper across its two copies. On a
corpus with near-duplicates this starves other documents of retrieval slots and
makes synthesis harder. Ingestion dedups by content hash, which by design does not
catch the same paper in two formats.

### Observation — figure captioning silently drops many figures
**29 empty vision responses** across the corpus (QpiGNN alone lost ~9 figures).
The graceful-skip path works, but dense academic figures frequently return empty
from `deepseek-v4-flash-vision-exp`, so figure content is under-represented.

### Observation — Gate 2 now dominates latency
`gate2` p95 is **13.1 s**, roughly 5× `generate`. It runs one claim-extraction call
plus an NLI scan per claim across contributing chunks, so cost scales with claim
count — the same over-decomposition behind Bug 1.

## What this says about the fixes under test

- **Fix 1 (doc-scoped retrieval / summarization)**: no regressions observed; no
  question named a filename, so the resolver correctly stayed inert.
- **Fix 2 (external scoping)**: every query ran `corpus_only=True` with zero
  external content retrieved. The context-adherence half is clearly working —
  Q13, Q18 and Q21 all declined honestly instead of filling gaps from background
  knowledge, which is exactly the failure Fix 2 targeted. The Gate 1 half appears
  over-tightened (Bug 2).
- **Claim-extraction fix**: no truncation or mega-claim collapse anywhere in 21
  questions (that fallback never fired). It did expose that fine-grained,
  faithful decomposition makes Bug 1 worse, since more claims means more chances
  for one to fail.
- **False-premise stress tests**: both correct (Q20 abstained, Q21 declined
  explicitly), and the Q19 corpus-level question correctly abstained. Gate 1 was
  not loosened.
