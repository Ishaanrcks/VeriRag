# MMLongBench-Doc — mmlongbench_subset200_v1canonical_tier1 — 2026-10-31

36 questions scored · scope `pooled` · extractor `deepseek-chat` · 164 skipped (document not in store) · store `/vol/mmlongbench`

## STRICT (headline): extraction restricted to the Verified facts

| accuracy | F1 | recall | precision |
|---|---|---|---|
| **0.250** | **0.167** | 0.100 | 0.500 |

## LENIENT (upper bound, includes unverified statements)

| accuracy | F1 | recall | precision |
|---|---|---|---|
| 0.278 | 0.222 | 0.133 | 0.667 |

## Outcomes

| outcome | count |
|---|---|
| answered (fully verified) | 8 |
| partial (some claims unverified) | 9 |
| declined | 19 |

| metric | value |
|---|---|
| recall (answered in full) | 26.7% |
| recall incl. partial (upper bound) | 46.7% |
| false positive rate (any answer where a decline was expected) | 50.0% |
| false positive rate, verified answers only | 0.0% |
| partial on should-decline (accepted tradeoff) | 3 |
| partial on should-answer | 6 |
| refusal precision | 15.8% |

## By answerability

| slice | n | strict acc | lenient acc |
|---|---|---|---|
| cross_page | 12 | 0.083 | 0.083 |
| single_page | 18 | 0.111 | 0.167 |
| unanswerable | 6 | 1.000 | 1.000 |

## By evidence_source

| slice | n | strict acc | lenient acc |
|---|---|---|---|
| (none) | 7 | 0.857 | 0.857 |
| chart | 5 | 0.000 | 0.000 |
| figure | 6 | 0.000 | 0.000 |
| layout | 3 | 0.333 | 0.333 |
| table | 12 | 0.167 | 0.250 |
| text | 11 | 0.000 | 0.000 |

## By doc_type

| slice | n | strict acc | lenient acc |
|---|---|---|---|
| Academic paper | 9 | 0.111 | 0.222 |
| Administration/Industry file | 4 | 0.250 | 0.250 |
| Brochure | 4 | 0.250 | 0.250 |
| Financial report | 12 | 0.250 | 0.250 |
| Research report / Introduction | 5 | 0.600 | 0.600 |
| Tutorial/Workshop | 2 | 0.000 | 0.000 |

## By group

| slice | n | strict acc | lenient acc |
|---|---|---|---|
| chart | 2 | 0.000 | 0.000 |
| figure | 4 | 0.000 | 0.000 |
| layout | 2 | 0.500 | 0.500 |
| multiple | 8 | 0.000 | 0.000 |
| table | 9 | 0.222 | 0.333 |
| text | 4 | 0.000 | 0.000 |
| unanswerable | 6 | 1.000 | 1.000 |
| unlabeled_evidence | 1 | 0.000 | 0.000 |

## By v2_status

| slice | n | strict acc | lenient acc |
|---|---|---|---|
| corrected | 2 | 1.000 | 1.000 |
| reworded | 4 | 0.000 | 0.000 |
| unchanged | 30 | 0.233 | 0.267 |

## By outcome

| slice | n | strict acc | lenient acc |
|---|---|---|---|
| abstained | 16 | 0.125 | 0.125 |
| answered | 8 | 0.250 | 0.250 |
| partial | 9 | 0.444 | 0.556 |
| refused | 3 | 0.333 | 0.333 |

## Per question

| id | stratum | want | outcome | strict | lenient |
|---|---|---|---|---|---|
| 1 | multiple | answer | refused | 0.00 | 0.00 |
| 2 | table | answer | partial | 0.00 | 0.00 |
| 8 | unanswerable | decline | partial | 1.00 | 1.00 |
| 20 | multiple | answer | answered | 0.00 | 0.00 |
| 23 | multiple | answer | abstained | 0.00 | 0.00 |
| 29 | table | answer | partial | 1.00 | 1.00 |
| 30 | table | answer | partial | 0.00 | 0.00 |
| 43 | figure | answer | refused | 0.00 | 0.00 |
| 45 | text | answer | partial | 0.00 | 0.00 |
| 50 | unanswerable | decline | abstained | 1.00 | 1.00 |
| 54 | figure | answer | abstained | 0.00 | 0.00 |
| 55 | chart | answer | abstained | 0.00 | 0.00 |
| 63 | text | answer | abstained | 0.00 | 0.00 |
| 64 | table | answer | answered | 1.00 | 1.00 |
| 75 | unanswerable | decline | refused | 1.00 | 1.00 |
| 78 | multiple | answer | answered | 0.00 | 0.00 |
| 82 | multiple | answer | abstained | 0.00 | 0.00 |
| 84 | table | answer | abstained | 0.00 | 0.00 |
| 86 | multiple | answer | answered | 0.00 | 0.00 |
| 87 | unanswerable | decline | partial | 1.00 | 1.00 |
| 93 | table | answer | abstained | 0.00 | 0.00 |
| 96 | unanswerable | decline | partial | 1.00 | 1.00 |
| 113 | table | answer | abstained | 0.00 | 0.00 |
| 120 | layout | answer | answered | 1.00 | 1.00 |
| 133 | unanswerable | decline | abstained | 1.00 | 1.00 |
| 135 | chart | answer | answered | 0.00 | 0.00 |
| 136 | table | answer | partial | 0.00 | 1.00 |
| 146 | multiple | answer | answered | 0.00 | 0.00 |
| 152 | table | answer | abstained | 0.00 | 0.00 |
| 157 | multiple | answer | abstained | 0.00 | 0.00 |
| 172 | layout | answer | abstained | 0.00 | 0.00 |
| 184 | figure | answer | abstained | 0.00 | 0.00 |
| 186 | text | answer | answered | 0.00 | 0.00 |
| 190 | text | answer | partial | 0.00 | 0.00 |
| 196 | unlabeled_evidence | answer | abstained | 0.00 | 0.00 |
| 197 | figure | answer | abstained | 0.00 | 0.00 |