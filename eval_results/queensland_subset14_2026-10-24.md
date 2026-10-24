# Eval: queensland_subset14 — 2026-10-24

Store: `eval_store/queensland` · 6 documents · 14 questions · `corpus_only=True`

**Hallucination rate: 0.0%** — no scored answer was factually wrong.

## Metrics

| metric | value |
|---|---|
| questions | 14 |
| answered | 8 |
| declined | 6 |
| **recall** (answerable actually answered) | **70.0%** |
| **refusal precision** (declines that were right) | **50.0%** |
| **false positive rate** (should decline, answered) | **25.0%** |
| **hallucination rate** (scored answers wrong) | **0.0%** (7 scored) |
| avg regenerations | 0.79 |

## By stratum

| stratum | n | answered | declined | wrong |
|---|---|---|---|---|
| false_premise | 2 | 0 | 2 | 0 |
| single | 8 | 7 | 1 | 0 |
| synthesis | 4 | 1 | 3 | 0 |

## Before / after vs baseline

Baseline: `baseline` (2026-10-23)

| Q | stratum | before | after | change |
|---|---|---|---|---|
| 1 | single | answered | answered | unchanged |
| 2 | single | answered | answered | unchanged |
| 3 | single | answered | answered | unchanged |
| 4 | single | answered | answered | unchanged |
| 6 | single | abstained | abstained | unchanged |
| 8 | single | answered | answered | unchanged |
| 9 | single | answered | answered | unchanged |
| 11 | single | answered | answered | unchanged |
| 13 | synthesis | refused | abstained | changed |
| 16 | synthesis | abstained | answered | now answers |
| 18 | synthesis | abstained | abstained | unchanged |
| 19 | synthesis | abstained | abstained | unchanged |
| 20 | false_premise | refused | refused | unchanged |
| 21 | false_premise | refused | refused | unchanged |

## Per question

| Q | stratum | want | outcome | correct | regens | meets |
|---|---|---|---|---|---|---|
| 1 | single | answer | answered | yes | 0 | ✅ |
| 2 | single | answer | answered | yes | 0 | ✅ |
| 3 | single | answer | answered | yes | 0 | ✅ |
| 4 | single | answer | answered | yes | 0 | ✅ |
| 6 | single | answer | abstained | — | 2 | ❌ |
| 8 | single | answer | answered | yes | 1 | ✅ |
| 9 | single | answer | answered | yes | 1 | ✅ |
| 11 | single | answer | answered | yes | 1 | ✅ |
| 13 | synthesis | answer | abstained | — | 2 | ❌ |
| 16 | synthesis | decline | answered | — | 1 | ❌ |
| 18 | synthesis | answer | abstained | — | 2 | ❌ |
| 19 | synthesis | decline | abstained | — | 0 | ✅ |
| 20 | false_premise | decline | refused | — | 0 | ✅ |
| 21 | false_premise | decline | refused | — | 1 | ✅ |

## Mismatches

**Q6** (single, wanted answer, got abstained)

- question: What two model families does ExpressGNN combine, and what does it use graph neural networks for?
- ground truth: Markov Logic Networks and GNNs; GNNs perform variational inference in the MLN.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- ExpressGNN uses graph neural networks for variational inference in Markov Logic Networks.

**Q13** (synthesis, wanted answer, got abstained)

- question: How does G-reasoner's parameter count compare with GFM-RAG's, and what limitation of GFM-RAG does G-reasoner set out to address?
- ground truth: 34M vs 8M. G-reasoner states GFM-RAG is limited to specific knowledge graphs and cannot generalize to other graph structures.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- G-reasoner introduces a 34M-parameter graph foundation model that jointly captures graph topology and textual semantics.
- GFM-RAG's GFM has 8M parameters.

**Q16** (synthesis, wanted decline, got answered)

- question: Which papers in this corpus apply graph neural networks, and to what different problems does each apply them?
- ground truth: GFM-RAG and G-reasoner (retrieval for RAG), ExpressGNN (probabilistic logic inference in MLNs), QpiGNN (uncertainty quantification for node regression).
- system: Based on the provided passages, three papers describe work involving graph neural networks. ExpressGNN uses a GNN variant for variational inference in Markov Logic Networks, aimed at efficient probabilistic logic reasoning over knowledge graphs [1]. GFM-RAG is described as powered by a graph neural 

**Q18** (synthesis, wanted answer, got abstained)

- question: Do GFM-RAG and G-reasoner share any authors?
- ground truth: Yes - Linhao Luo, Zicheng Zhao, Chen Gong, Gholamreza Haffari, Dinh Phung and Shirui Pan appear on both.
- system: Could not produce a fully grounded answer after 2 regeneration attempt(s). These claims could not be verified against the sources:
- The GFM-RAG paper lists Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Chen Gong, Dinh Phung, and Shirui Pan as authors.
- The G-reasoner paper lists Linhao Luo, Zichen
