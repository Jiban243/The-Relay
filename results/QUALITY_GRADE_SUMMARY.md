# The Relay — Quality Audit

Method: existing human grades preserved; remaining responses screened using a frozen offline answer-concept and citation rubric.

This is not independent expert verification of every factual claim.

| Setting | N | Correct and cited | Wrong incl. wrongful refusal | Correctly refused | Avg. tokens |
|---|---:|---:|---:|---:|---:|
| small | 15 | 73.3% | 6.7% | 20.0% | 1110.9 |
| balanced | 15 | 73.3% | 6.7% | 20.0% | 2879.6 |
| reranked | 15 | 73.3% | 6.7% | 20.0% | 3085.3 |

## Individual decisions

| Question | Setting | Grade | Method | Reason |
|---|---|---|---|---|
| a1 | small | correct | human_existing | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| a1 | balanced | correct | human_existing | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| a1 | reranked | correct | human_existing | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| a2 | balanced | refused_wrongly | human_existing | Refused a question answerable from the papers |
| a2 | reranked | refused_wrongly | human_existing | Refused a question answerable from the papers |
| a2 | small | refused_wrongly | human_existing | Refused a question answerable from the papers |
| a3 | reranked | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| a3 | small | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| a3 | balanced | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| a4 | small | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| a4 | balanced | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| a4 | reranked | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| m1 | balanced | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| m1 | reranked | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| m1 | small | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| m2 | reranked | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| m2 | small | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| m2 | balanced | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| m3 | small | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| m3 | balanced | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| m3 | reranked | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| m4 | balanced | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| m4 | reranked | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| m4 | small | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| h1 | reranked | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| h1 | small | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| h1 | balanced | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| h2 | small | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| h2 | balanced | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| h2 | reranked | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| h3 | balanced | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| h3 | reranked | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| h3 | small | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| h4 | reranked | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| h4 | small | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| h4 | balanced | correct | offline_rubric_v1 | Core concepts and citation provenance passed the offline rubric; additional unsupported claims are not independently ruled out |
| u1 | small | refused_correctly | offline_rubric_v1 | Correctly refused an out-of-library question |
| u1 | balanced | refused_correctly | offline_rubric_v1 | Correctly refused an out-of-library question |
| u1 | reranked | refused_correctly | offline_rubric_v1 | Correctly refused an out-of-library question |
| u2 | balanced | refused_correctly | offline_rubric_v1 | Correctly refused an out-of-library question |
| u2 | reranked | refused_correctly | offline_rubric_v1 | Correctly refused an out-of-library question |
| u2 | small | refused_correctly | offline_rubric_v1 | Correctly refused an out-of-library question |
| u3 | reranked | refused_correctly | offline_rubric_v1 | Correctly refused an out-of-library question |
| u3 | small | refused_correctly | offline_rubric_v1 | Correctly refused an out-of-library question |
| u3 | balanced | refused_correctly | offline_rubric_v1 | Correctly refused an out-of-library question |

Limitation: lexical concept checks can miss unsupported additional claims and can reject valid paraphrases. The scores are rubric-based.
