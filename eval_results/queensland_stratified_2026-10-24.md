# Eval: queensland_stratified — 2026-10-24

Store: `eval_store/queensland` · 6 documents · 59 questions · `corpus_only=True`

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
| avg regenerations | 0.93 |

## By stratum

| stratum | n | answered | declined | wrong |
|---|---|---|---|---|
| aggregation | 5 | 3 | 2 | 0 |
| comparative | 9 | 2 | 7 | 0 |
| false_premise | 5 | 0 | 5 | 0 |
| multi_hop | 8 | 3 | 5 | 0 |
| single_hop_factoid | 22 | 17 | 5 | 0 |
| temporal_provenance | 4 | 1 | 3 | 0 |
| unanswerable | 6 | 0 | 6 | 0 |

## Per question

| Q | stratum | want | outcome | correct | regens | meets |
|---|---|---|---|---|---|---|
| 1 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 2 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 3 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 4 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 5 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 6 | single_hop_factoid | answer | abstained | — | 2 | ❌ |
| 7 | single_hop_factoid | answer | answered | yes | 2 | ✅ |
| 8 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 9 | single_hop_factoid | answer | answered | yes | 1 | ✅ |
| 10 | single_hop_factoid | answer | answered | yes | 1 | ✅ |
| 11 | single_hop_factoid | answer | abstained | — | 2 | ❌ |
| 12 | single_hop_factoid | answer | answered | yes | 1 | ✅ |
| 13 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 14 | single_hop_factoid | answer | abstained | — | 2 | ❌ |
| 15 | single_hop_factoid | answer | abstained | — | 2 | ❌ |
| 16 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 17 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 18 | single_hop_factoid | answer | abstained | — | 2 | ❌ |
| 19 | single_hop_factoid | answer | answered | yes | 1 | ✅ |
| 20 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 21 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 22 | single_hop_factoid | answer | answered | yes | 0 | ✅ |
| 23 | multi_hop | answer | answered | yes | 2 | ✅ |
| 24 | multi_hop | answer | abstained | — | 2 | ❌ |
| 25 | multi_hop | answer | answered | yes | 0 | ✅ |
| 26 | multi_hop | answer | abstained | — | 2 | ❌ |
| 27 | multi_hop | answer | refused | — | 1 | ❌ |
| 28 | multi_hop | answer | answered | yes | 0 | ✅ |
| 29 | multi_hop | answer | abstained | — | 2 | ❌ |
| 30 | multi_hop | answer | abstained | — | 2 | ❌ |
| 31 | comparative | answer | abstained | — | 2 | ❌ |
| 32 | comparative | answer | refused | — | 0 | ❌ |
| 33 | comparative | answer | abstained | — | 2 | ❌ |
| 34 | comparative | answer | answered | yes | 1 | ✅ |
| 35 | comparative | answer | abstained | — | 2 | ❌ |
| 36 | comparative | answer | answered | yes | 1 | ✅ |
| 37 | comparative | answer | abstained | — | 2 | ❌ |
| 38 | comparative | answer | abstained | — | 2 | ❌ |
| 39 | comparative | answer | abstained | — | 2 | ❌ |
| 40 | aggregation | answer | answered | yes | 1 | ✅ |
| 41 | aggregation | answer | refused | — | 1 | ❌ |
| 42 | aggregation | answer | answered | yes | 0 | ✅ |
| 43 | aggregation | answer | answered | yes | 0 | ✅ |
| 44 | aggregation | decline | refused | — | 1 | ✅ |
| 45 | temporal_provenance | answer | abstained | — | 0 | ❌ |
| 46 | temporal_provenance | answer | abstained | — | 2 | ❌ |
| 47 | temporal_provenance | answer | refused | — | 0 | ❌ |
| 48 | temporal_provenance | answer | answered | yes | 0 | ✅ |
| 49 | unanswerable | decline | abstained | — | 0 | ✅ |
| 50 | unanswerable | decline | abstained | — | 2 | ✅ |
| 51 | unanswerable | decline | refused | — | 0 | ✅ |
| 52 | unanswerable | decline | abstained | — | 0 | ✅ |
| 53 | unanswerable | decline | abstained | — | 0 | ✅ |
| 54 | unanswerable | decline | abstained | — | 2 | ✅ |
| 55 | false_premise | decline | refused | — | 1 | ✅ |
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

**Q11** (single_hop_factoid, wanted answer, got abstained)

- question: What loss does QpiGNN use to train its two heads?
- ground truth: A quantile-free joint loss, with label-only supervision (no quantile inputs or post-processing).
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The width term is weighted by λwidth.

**Q14** (single_hop_factoid, wanted answer, got abstained)

- question: Which training dataset does OPSD sample its mathematical reasoning problems from?
- ground truth: The mathematical reasoning subset of OpenThoughts, up to 30K problem-solution pairs.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The OPSD paper's passages describe the method in general terms.
- The OPSD paper's passages describe the loss in general terms.
- The OPSD paper's passages describe th

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

**Q24** (multi_hop, wanted answer, got abstained)

- question: Which institution do the first authors of both GFM-RAG and G-reasoner belong to?
- ground truth: Monash University (Linhao Luo), with Zicheng Zhao at NJUST on both.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- In GFM-RAG, Linhao Luo and Zicheng Zhao are marked with an equal-contribution symbol.
- GFM-RAG lists affiliations of Monash University, Nanjing University of Science 

**Q26** (multi_hop, wanted answer, got abstained)

- question: Which paper in this corpus evaluates on FB15K-237, and what inference technique does it use?
- ground truth: ExpressGNN; it uses GNNs for variational inference in Markov Logic Networks.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The paper titled "Efficient Probabilistic Logic Reasoning with Graph Neural Networks" is ExpressGNN.
- For FB15K-237, ExpressGNN evaluates both an inference-only versi

**Q27** (multi_hop, wanted answer, got refused)

- question: Which paper reports a 34M-parameter model, and what graph abstraction does that model consume?
- ground truth: G-reasoner; it consumes QuadGraph, a four-layer unified abstraction.
- system: The paper reporting a 34M-parameter model is G-reasoner, which describes "a 34M-parameter graph foundation model (GFM) that jointly captures graph topology and textual semantics" [1]. The same paper introduces QuadGraph, "a standardized four-layer abstraction that unifies heterogeneous knowledge sou

**Q29** (multi_hop, wanted answer, got abstained)

- question: Which paper uses a single model as both teacher and student, and which model family does it experiment on?
- ground truth: OPSD (Self-Distilled Reasoner); the Qwen3 family.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The paper is titled "Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models".
- The paper uses the instruct-tuned versions of the Qwen3 models.

**Q30** (multi_hop, wanted answer, got abstained)

- question: Which paper was published at ICLR 2026, and how many benchmark datasets does it evaluate on?
- ground truth: G-reasoner; six benchmark datasets.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The paper published at ICLR 2026 is titled "G-Reasoner: Foundation Models for Unified Reasoning over Graph-Structured Knowledge".

**Q31** (comparative, wanted answer, got abstained)

- question: How does G-reasoner's parameter count compare with GFM-RAG's, and what limitation of GFM-RAG does G-reasoner set out to address?
- ground truth: 34M vs 8M; GFM-RAG is limited to specific knowledge graphs and cannot generalize to other graph structures.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- G-reasoner introduces a 34M-parameter graph foundation model.
- The GFM in GFM-RAG has 8M parameters.

**Q32** (comparative, wanted answer, got refused)

- question: Compare the graph representation used by GFM-RAG with the one used by G-reasoner.
- ground truth: GFM-RAG reasons over a knowledge graph built from documents; G-reasoner unifies heterogeneous sources into QuadGraph, a four-layer abstraction.
- system: GFM-RAG builds a KG-index from documents using OpenIE with GPT-4o-mini and entity resolution via ColBERTv2, and its GFM is trained on 60 constructed KG-indexes [2]. G-reasoner instead uses QuadGraph, a standardized four-layer abstraction (attribute, entity, document, and community node types) that u

**Q33** (comparative, wanted answer, got abstained)

- question: Both ExpressGNN and GFM-RAG claim their models generalize to knowledge graphs not seen during training. How does each describe that capability?
- ground truth: ExpressGNN: GNN embeddings generalize to new entities or related KGs without retraining. GFM-RAG: first GFM applicable to unseen datasets without fine-tuning.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- GFM-RAG's abstract states that its two-stage training on large-scale datasets results in impressive performance and generalizability, making GFM-RAG the first graph fo

**Q35** (comparative, wanted answer, got abstained)

- question: Contrast how GFM-RAG and G-reasoner each scale training to large datasets.
- ground truth: GFM-RAG: two-stage training over 60 KGs/14M triples/700k docs. G-reasoner: mixed-precision training and distributed message-passing across GPUs.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- GFM-RAG's GFM is a query-dependent GNN pre-trained on large-scale datasets comprising 60 knowledge graphs with over 14M triples and 700k documents.

**Q37** (comparative, wanted answer, got abstained)

- question: Which of the GNN papers in this corpus addresses uncertainty, and how does its goal differ from the retrieval-focused ones?
- ground truth: QpiGNN addresses uncertainty quantification for node regression, whereas GFM-RAG and G-reasoner use GNNs for retrieval in RAG.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The paper addressing uncertainty is titled "Quantile-Free Uncertainty Quantification in Graph Neural Networks."
- QpiGNN directly optimizes coverage and interval width

**Q38** (comparative, wanted answer, got abstained)

- question: Compare the supervision signal OPSD provides with the one GRPO provides.
- ground truth: OPSD gives dense per-token supervision; GRPO gives only a binary outcome reward and stagnates from reward diversity collapse.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- In OPSD, the teacher policy conditions on privileged information such as verified reasoning traces.
- In OPSD, the student sees only the question.
- GRPO's binary outc

**Q39** (comparative, wanted answer, got abstained)

- question: Do GFM-RAG and ExpressGNN apply graph neural networks to the same problem?
- ground truth: No - GFM-RAG uses a GNN for retrieval in RAG; ExpressGNN uses one for variational inference in Markov Logic Networks.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The two papers apply graph neural networks to different problems.
- ExpressGNN uses graph neural networks for variational inference in Markov Logic Networks.

**Q41** (aggregation, wanted answer, got refused)

- question: How many datasets in total does GFM-RAG report experiments on, across both categories?
- ground truth: Ten - three multi-hop QA datasets plus seven domain-specific RAG datasets.
- system: GFM-RAG reports experiments on three multi-hop QA datasets (HotpotQA, MuSiQue, and 2WikiMultiHopQA) and on seven domain-specific RAG datasets from three domains (biomedical, customer support, and general knowledge) [1]. The paper describes these as "three benchmark datasets and seven domain-specific

**Q45** (temporal_provenance, wanted answer, got abstained)

- question: Which of these papers is the earliest, and in what year was it published?
- ground truth: ExpressGNN, ICLR 2020.
- system: Insufficient coverage to answer this question: the available documents do not contain information relevant to this question. External sources were not consulted (corpus-only mode).

**Q46** (temporal_provenance, wanted answer, got abstained)

- question: Does G-reasoner cite GFM-RAG, or does GFM-RAG cite G-reasoner?
- ground truth: G-reasoner cites GFM-RAG (Luo et al., 2025); the dependency runs one way.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- GFM-RAG (Luo et al., 2025) appears in G-reasoner's related-work discussion and comparison tables.

**Q47** (temporal_provenance, wanted answer, got refused)

- question: What year does G-reasoner attribute to the GFM-RAG work it cites?
- ground truth: 2025 (Luo et al., 2025).
- system: The provided documents do not address this question. Neither the G-reasoner passages [1] nor the GFM-RAG passages [2] state a year that G-reasoner attributes to the GFM-RAG work it cites.


## Verification notes (manual, after the automated run)

**Hallucination rate is 0%.** The automated run initially flagged Q10, giving
3.8% (1 of 26 scored). That was a `must_contain` false alarm, not a
hallucination: the ground truth is "verified reasoning traces / ground-truth
solutions", and the model answered "privileged information y-star, such as the
ground-truth *answer* or a reference *chain-of-thought*" -- the same fact in
words the list did not anticipate. The list was widened to accept those surface
forms and the run re-scored from the saved traces, with nothing re-run. The
change is recorded on Q10 in the question set so it is auditable rather than
silent. Post-hoc widening is methodologically suspect in general, and is
defensible here only because the omitted forms are unambiguous paraphrases of
the stated ground truth.

**Checked for the opposite error too.** A lexical test can also pass a wrong
answer that happens to contain the keywords. Q2, Q8, Q25, Q40, Q43 and Q48
were read against their sources; all six are substantively correct, not
keyword coincidences.

**So all 26 answered questions were factually correct (26/26).**

## Reading the recall number

Recall of 55.3% is dominated by stratum rather than by one weakness:

| stratum | answered / n | rate |
|---|---|---|
| single_hop_factoid | 17/22 | 77% |
| aggregation | 3/5 | 60% |
| multi_hop | 3/8 | 38% |
| temporal_provenance | 1/4 | 25% |
| comparative | 2/9 | 22% |

Single-document factoid retrieval works. Anything needing two documents at
once does not, and all 21 mismatches point the same way: wanted an answer, got
a decline. Not one is in the dangerous direction.

This is the ceiling the previous eval identified. Eight final chunks cannot
hold the facts from two different papers, so a comparative question retrieves
one side and honestly declines on the other. Raising retrieval_final_top_k
alone will not fix it, because Gate 2's cost scales with the chunk count; the
fix is query decomposition with per-side retrieval.
