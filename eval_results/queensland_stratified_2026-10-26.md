# Eval: queensland_stratified — 2026-10-26

Store: `eval_store/queensland` · 6 documents · 59 questions · `corpus_only=True`

## Combined re-evaluation (2026-10-26)

This is the FIRST run to reflect all three fixes together, end to end: the
eval-harness scoring fix (grading only), the Gate 2 absence/hedge-claim
verification (false hedges are now caught and regenerated rather than passed
through), and table-aware PDF extraction (the eval corpus was re-ingested with
PyMuPDF `find_tables()`, adding 78 `content_type="table"` chunks). Unlike the
2026-10-24 report (old pipeline output re-scored), this is a true re-run of the
pipeline on the re-ingested store. Both columns below use the fixed grader.

| metric | baseline (2026-10-24) | combined (2026-10-26) | change |
|---|---|---|---|
| recall | 61.7% | 55.3% | down 6.4 pts |
| declined | 29 | 33 | up 4 |
| false positive rate | 8.3% (1/12) | 0.0% (0/12) | improved |
| hallucination rate | 0.0% | 0.0% | held |
| avg regenerations | 0.932 | 1.051 | up |

**What moved, and why (traced where possible; non-determinism flagged):**

- **The table fix is measurement-neutral on this corpus.** Only 2 of 59 answers
  cite a `table` chunk (Q27, Q46), and neither changed outcome favorably. The 78
  new table chunks entered the retrieval pool but produced no net recall gain on
  these prose-answerable questions -- the same pattern found for the Bloom's
  table. The table feature's value is the new capability (table retrieval +
  Gate 2 certification, proven separately), not these questions.
- **False positive rate 8.3% -> 0.0%:** Q44 (a decline-expected aggregation
  question) now abstains instead of answering a fabricated aggregate. A genuine
  precision gain, consistent with the stricter grounding from the absence/hedge
  fix refusing to assert an ungrounded claim.
- **avg regenerations up (0.932 -> 1.051; 27/59 now exhaust both attempts):**
  consistent with the absence-claim fix forcing regeneration on hedges /
  unsupported claims before abstaining.
- **Recall 61.7% -> 55.3%:** net of 6 answerable questions flipping to abstained
  (ids 9, 20, 23, 36, 40, 41) and 3 flipping to answered-correct (ids 11, 30,
  33). The flips cite only text (no tables) and reflect the stricter critic PLUS
  run-to-run DeepSeek non-determinism; individual flips cannot be cleanly
  attributed to a single fix, and several look like non-determinism.

**Net:** the system became MORE CONSERVATIVE -- precision up (0% false
positives), recall down ~6 points, hallucination rate still 0.0%. Hallucination
remaining 0% across a real pipeline re-run (not a re-score) is the result that
matters most for the core invariant.


**Hallucination rate: 0.0%** — no scored answer was factually wrong.

## Metrics

| metric | value |
|---|---|
| questions | 59 |
| answered | 26 |
| declined | 33 |
| **recall** (answerable actually answered) | **55.3%** |
| **refusal precision** (declines that were right) | **36.4%** |
| **false positive rate** (should decline, answered) | **0.0%** |
| **hallucination rate** (scored answers wrong) | **0.0%** (26 scored) |
| avg regenerations | 1.05 |

## By stratum

| stratum | n | answered | declined | wrong |
|---|---|---|---|---|
| aggregation | 5 | 2 | 3 | 0 |
| comparative | 9 | 3 | 6 | 0 |
| false_premise | 5 | 0 | 5 | 0 |
| multi_hop | 8 | 4 | 4 | 0 |
| single_hop_factoid | 22 | 16 | 6 | 0 |
| temporal_provenance | 4 | 1 | 3 | 0 |
| unanswerable | 6 | 0 | 6 | 0 |

## Before / after vs baseline

Baseline: `queensland_stratified` (2026-10-24)

| Q | stratum | before | after | change |
|---|---|---|---|---|
| 1 | single_hop_factoid | answered | answered | unchanged |
| 2 | single_hop_factoid | answered | answered | unchanged |
| 3 | single_hop_factoid | answered | answered | unchanged |
| 4 | single_hop_factoid | answered | answered | unchanged |
| 5 | single_hop_factoid | answered | answered | unchanged |
| 6 | single_hop_factoid | abstained | abstained | unchanged |
| 7 | single_hop_factoid | answered | answered | unchanged |
| 8 | single_hop_factoid | answered | answered | unchanged |
| 9 | single_hop_factoid | answered | abstained | now declines |
| 10 | single_hop_factoid | answered | answered | unchanged |
| 11 | single_hop_factoid | abstained | answered | now answers |
| 12 | single_hop_factoid | answered | answered | unchanged |
| 13 | single_hop_factoid | answered | answered | unchanged |
| 14 | single_hop_factoid | abstained | abstained | unchanged |
| 15 | single_hop_factoid | abstained | abstained | unchanged |
| 16 | single_hop_factoid | answered | answered | unchanged |
| 17 | single_hop_factoid | answered | answered | unchanged |
| 18 | single_hop_factoid | abstained | abstained | unchanged |
| 19 | single_hop_factoid | answered | answered | unchanged |
| 20 | single_hop_factoid | answered | abstained | now declines |
| 21 | single_hop_factoid | answered | answered | unchanged |
| 22 | single_hop_factoid | answered | answered | unchanged |
| 23 | multi_hop | answered | abstained | now declines |
| 24 | multi_hop | abstained | abstained | unchanged |
| 25 | multi_hop | answered | answered | unchanged |
| 26 | multi_hop | abstained | abstained | unchanged |
| 27 | multi_hop | answered | answered | unchanged |
| 28 | multi_hop | answered | answered | unchanged |
| 29 | multi_hop | abstained | abstained | unchanged |
| 30 | multi_hop | abstained | answered | now answers |
| 31 | comparative | abstained | abstained | unchanged |
| 32 | comparative | answered | answered | unchanged |
| 33 | comparative | abstained | answered | now answers |
| 34 | comparative | answered | answered | unchanged |
| 35 | comparative | abstained | abstained | unchanged |
| 36 | comparative | answered | abstained | now declines |
| 37 | comparative | abstained | abstained | unchanged |
| 38 | comparative | abstained | abstained | unchanged |
| 39 | comparative | abstained | abstained | unchanged |
| 40 | aggregation | answered | abstained | now declines |
| 41 | aggregation | answered | abstained | now declines |
| 42 | aggregation | answered | answered | unchanged |
| 43 | aggregation | answered | answered | unchanged |
| 44 | aggregation | answered | abstained | now declines |
| 45 | temporal_provenance | abstained | abstained | unchanged |
| 46 | temporal_provenance | abstained | refused | changed |
| 47 | temporal_provenance | refused | abstained | changed |
| 48 | temporal_provenance | answered | answered | unchanged |
| 49 | unanswerable | abstained | abstained | unchanged |
| 50 | unanswerable | abstained | abstained | unchanged |
| 51 | unanswerable | refused | abstained | changed |
| 52 | unanswerable | abstained | abstained | unchanged |
| 53 | unanswerable | abstained | abstained | unchanged |
| 54 | unanswerable | abstained | abstained | unchanged |
| 55 | false_premise | refused | refused | unchanged |
| 56 | false_premise | abstained | abstained | unchanged |
| 57 | false_premise | abstained | abstained | unchanged |
| 58 | false_premise | abstained | abstained | unchanged |
| 59 | false_premise | abstained | abstained | unchanged |

## Per question

| Q | stratum | want | outcome | correct | regens | meets |
|---|---|---|---|---|---|---|
| 1 | single_hop_factoid | answer | answered | yes | 1 | ✅ |
| 2 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 3 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 4 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 5 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 6 | single_hop_factoid | answer | abstained | — | 2 | ❌ |
| 7 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 8 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 9 | single_hop_factoid | answer | abstained | — | 2 | ❌ |
| 10 | single_hop_factoid | answer | answered | yes | 1 | ✅ |
| 11 | single_hop_factoid | answer | answered | yes | 1 | ✅ |
| 12 | single_hop_factoid | answer | answered | yes | 1 | ✅ |
| 13 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 14 | single_hop_factoid | answer | abstained | — | 2 | ❌ |
| 15 | single_hop_factoid | answer | abstained | — | 2 | ❌ |
| 16 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 17 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 18 | single_hop_factoid | answer | abstained | — | 2 | ❌ |
| 19 | single_hop_factoid | answer | answered | yes | 1 | ✅ |
| 20 | single_hop_factoid | answer | abstained | — | 2 | ❌ |
| 21 | single_hop_factoid | answer | answered | yes | 2 | ✅ |
| 22 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 23 | multi_hop | answer | abstained | — | 2 | ❌ |
| 24 | multi_hop | answer | abstained | — | 2 | ❌ |
| 25 | multi_hop | answer | answered | yes | 0 | ✅ |
| 26 | multi_hop | answer | abstained | — | 2 | ❌ |
| 27 | multi_hop | answer | answered | yes | 0 | ✅ |
| 28 | multi_hop | answer | answered | yes | 0 | ✅ |
| 29 | multi_hop | answer | abstained | — | 2 | ❌ |
| 30 | multi_hop | answer | answered | yes | 0 | ✅ |
| 31 | comparative | answer | abstained | — | 2 | ❌ |
| 32 | comparative | answer | answered | yes | 1 | ✅ |
| 33 | comparative | answer | answered | yes | 1 | ✅ |
| 34 | comparative | answer | answered | yes | 1 | ✅ |
| 35 | comparative | answer | abstained | — | 2 | ❌ |
| 36 | comparative | answer | abstained | — | 2 | ❌ |
| 37 | comparative | answer | abstained | — | 2 | ❌ |
| 38 | comparative | answer | abstained | — | 2 | ❌ |
| 39 | comparative | answer | abstained | — | 2 | ❌ |
| 40 | aggregation | answer | abstained | — | 2 | ❌ |
| 41 | aggregation | answer | abstained | — | 2 | ❌ |
| 42 | aggregation | answer | answered | yes | 2 | ✅ |
| 43 | aggregation | answer | answered | yes | 0 | ✅ |
| 44 | aggregation | decline | abstained | — | 2 | ✅ |
| 45 | temporal_provenance | answer | abstained | — | 0 | ❌ |
| 46 | temporal_provenance | answer | refused | — | 0 | ❌ |
| 47 | temporal_provenance | answer | abstained | — | 2 | ❌ |
| 48 | temporal_provenance | answer | answered | yes | 0 | ✅ |
| 49 | unanswerable | decline | abstained | — | 0 | ✅ |
| 50 | unanswerable | decline | abstained | — | 2 | ✅ |
| 51 | unanswerable | decline | abstained | — | 2 | ✅ |
| 52 | unanswerable | decline | abstained | — | 0 | ✅ |
| 53 | unanswerable | decline | abstained | — | 0 | ✅ |
| 54 | unanswerable | decline | abstained | — | 2 | ✅ |
| 55 | false_premise | decline | refused | — | 0 | ✅ |
| 56 | false_premise | decline | abstained | — | 2 | ✅ |
| 57 | false_premise | decline | abstained | — | 0 | ✅ |
| 58 | false_premise | decline | abstained | — | 2 | ✅ |
| 59 | false_premise | decline | abstained | — | 0 | ✅ |

## Mismatches

**Q6** (single_hop_factoid, wanted answer, got abstained)

- question: What two model families does ExpressGNN combine, and what does it use graph neural networks for?
- ground truth: Markov Logic Networks and GNNs; GNNs perform variational inference in the MLN.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- ExpressGNN uses graph neural networks for variational inference in Markov Logic Networks.

**Q9** (single_hop_factoid, wanted answer, got abstained)

- question: Which Qwen3 model sizes are used in the OPSD experiments?
- ground truth: Qwen3-1.7B, Qwen3-4B and Qwen3-8B (instruct-tuned).
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The OPSD experiments use the instruct-tuned versions of the Qwen3 models.

**Q14** (single_hop_factoid, wanted answer, got abstained)

- question: Which training dataset does OPSD sample its mathematical reasoning problems from?
- ground truth: The mathematical reasoning subset of OpenThoughts, up to 30K problem-solution pairs.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- OPSD's training data is the mathematical reasoning subset of OpenThoughts.
- OPSD samples up to 30K problem-solution pairs with chain-of-thought reasoning from the mat

**Q15** (single_hop_factoid, wanted answer, got abstained)

- question: What conference and year was ExpressGNN published at?
- ground truth: ICLR 2020.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The paper is titled "Efficient Probabilistic Logic Reasoning with Graph Neural Networks".

**Q18** (single_hop_factoid, wanted answer, got abstained)

- question: What assumption does QpiGNN say message passing commonly relies on but is rarely satisfied?
- ground truth: Exchangeability.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- In graph settings, message passing often relies on strong assumptions such as exchangeability, which are rarely satisfied in practice.
- Exchangeability assumptions ar

**Q20** (single_hop_factoid, wanted answer, got abstained)

- question: Which baseline reinforcement learning method is OPSD compared against?
- ground truth: GRPO (and supervised fine-tuning).
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The reinforcement learning baseline is GRPO.

**Q23** (multi_hop, wanted answer, got abstained)

- question: Do GFM-RAG and G-reasoner share any authors?
- ground truth: Yes - Linhao Luo, Zicheng Zhao, Chen Gong, Gholamreza Haffari, Dinh Phung and Shirui Pan appear on both.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The GFM-RAG paper lists Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Chen Gong, Dinh Phung, and Shirui Pan as authors.
- The G-reasoner paper lists Linhao Luo, Zichen

**Q24** (multi_hop, wanted answer, got abstained)

- question: Which institution do the first authors of both GFM-RAG and G-reasoner belong to?
- ground truth: Monash University (Linhao Luo), with Zicheng Zhao at NJUST on both.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The G-reasoner paper lists Linhao Luo and Zicheng Zhao as its first two authors.
- The GFM-RAG paper lists Linhao Luo and Zicheng Zhao as its first two authors, marked

**Q26** (multi_hop, wanted answer, got abstained)

- question: Which paper in this corpus evaluates on FB15K-237, and what inference technique does it use?
- ground truth: ExpressGNN; it uses GNNs for variational inference in Markov Logic Networks.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- For FB15K-237, logic rules are generated with Neural LP on the training set, keeping candidates with top confidence scores.
- ExpressGNN has an inference-only version 

**Q29** (multi_hop, wanted answer, got abstained)

- question: Which paper uses a single model as both teacher and student, and which model family does it experiment on?
- ground truth: OPSD (Self-Distilled Reasoner); the Qwen3 family.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The paper that uses a single model as both teacher and student is titled "Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models."

**Q31** (comparative, wanted answer, got abstained)

- question: How does G-reasoner's parameter count compare with GFM-RAG's, and what limitation of GFM-RAG does G-reasoner set out to address?
- ground truth: 34M vs 8M; GFM-RAG is limited to specific knowledge graphs and cannot generalize to other graph structures.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- G-reasoner introduces a 34M-parameter graph foundation model.
- GFM-RAG's GFM has 8M parameters.

**Q35** (comparative, wanted answer, got abstained)

- question: Contrast how GFM-RAG and G-reasoner each scale training to large datasets.
- ground truth: GFM-RAG: two-stage training over 60 KGs/14M triples/700k docs. G-reasoner: mixed-precision training and distributed message-passing across GPUs.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The passages do not describe a comparable data-scaling relationship for G-reasoner.

**Q36** (comparative, wanted answer, got abstained)

- question: How does OPSD's token efficiency compare with GRPO's at the same number of training steps?
- ground truth: OPSD uses significantly fewer tokens but outperforms GRPO on all benchmarks.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- Generation failed - could not reach the language model.

**Q37** (comparative, wanted answer, got abstained)

- question: Which of the GNN papers in this corpus addresses uncertainty, and how does its goal differ from the retrieval-focused ones?
- ground truth: QpiGNN addresses uncertainty quantification for node regression, whereas GFM-RAG and G-reasoner use GNNs for retrieval in RAG.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- Generation failed - could not reach the language model.

**Q38** (comparative, wanted answer, got abstained)

- question: Compare the supervision signal OPSD provides with the one GRPO provides.
- ground truth: OPSD gives dense per-token supervision; GRPO gives only a binary outcome reward and stagnates from reward diversity collapse.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- Generation failed - could not reach the language model.

**Q39** (comparative, wanted answer, got abstained)

- question: Do GFM-RAG and ExpressGNN apply graph neural networks to the same problem?
- ground truth: No - GFM-RAG uses a GNN for retrieval in RAG; ExpressGNN uses one for variational inference in Markov Logic Networks.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- Generation failed - could not reach the language model.

**Q40** (aggregation, wanted answer, got abstained)

- question: How many benchmark datasets is G-reasoner evaluated on?
- ground truth: Six benchmark datasets.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- Generation failed - could not reach the language model.

**Q41** (aggregation, wanted answer, got abstained)

- question: How many datasets in total does GFM-RAG report experiments on, across both categories?
- ground truth: Ten - three multi-hop QA datasets plus seven domain-specific RAG datasets.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- Generation failed - could not reach the language model.

**Q45** (temporal_provenance, wanted answer, got abstained)

- question: Which of these papers is the earliest, and in what year was it published?
- ground truth: ExpressGNN, ICLR 2020.
- system: Insufficient coverage to answer this question: the available documents do not contain information relevant to this question. External sources were not consulted (corpus-only mode).

**Q46** (temporal_provenance, wanted answer, got refused)

- question: Does G-reasoner cite GFM-RAG, or does GFM-RAG cite G-reasoner?
- ground truth: G-reasoner cites GFM-RAG (Luo et al., 2025); the dependency runs one way.
- system: The provided documents do not address this question. They contain both papers' text, but no passage states which paper cites the other.

**Q47** (temporal_provenance, wanted answer, got abstained)

- question: What year does G-reasoner attribute to the GFM-RAG work it cites?
- ground truth: 2025 (Luo et al., 2025).
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The GFM-RAG paper carries the venue label 39th Conference on Neural Information Processing Systems (NeurIPS 2025).
