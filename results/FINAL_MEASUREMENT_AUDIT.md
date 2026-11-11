# The Relay — Results Audit

Recomputed from selected run IDs in results.csv.

## Five stand-in load scenarios

| Scenario | N | OK | Timeout | Rate limited | Error | Peak queue | Model TTFT p50 | p95 | p99 | Total p50 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| steady | 8 | 8 | 0 | 0 | 0 | 0 | 904.26 | 960.95 | 969.44 | 998.01 | 1048.73 | 1050.15 |
| burst | 24 | 6 | 9 | 0 | 9 | 12 | 1664.64 | 2451.95 | 2513.62 | 1760.51 | 2602.71 | 2678.88 |
| sustained_overload | 48 | 10 | 37 | 0 | 1 | 7 | 1670.8 | 4167.02 | 4256.3 | 1780.03 | 4408.08 | 4495.8 |
| rpm_halved_initial | 24 | 9 | 8 | 0 | 7 | 12 | 1928.49 | 2566.6 | 2671.99 | 2010.85 | 2731.0 | 2833.65 |
| daily_exhausted | 12 | 3 | 0 | 9 | 0 | 0 | 1161.96 | 1301.64 | 1314.06 | 1249.42 | 1398.25 | 1411.48 |

Percentiles use successful responses only.
Stand-in tokens and latency are simulated.
The application buffers its response, so model TTFT is not user-visible TTFT.

## Controlled simulated RPM experiment

| Run | N | OK | Queue timed out | Arrival RPM |
| --- | --- | --- | --- | --- |
| rpm_control_30 | 24 | 24 | 0 | 133.32 |
| rpm_halved_15 | 24 | 15 | 9 | 133.32 |

## Real Groq retrieval evaluation

| Setting | N | Mean tokens/answer | Provider first-content p50 | p95 | Total p50 | p95 | Saved grades |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small | 15 | 1110.93 | 1309.75 | 13617.07 | 1358.71 | 14033.24 | 15/15 |
| balanced | 15 | 2879.6 | 1132.55 | 5624.59 | 1547.19 | 6300.15 | 15/15 |
| reranked | 15 | 3085.33 | 1224.7 | 5161.27 | 1587.82 | 5519.75 | 15/15 |

Evidence screening (not independent factual grading):

- small: {"cited_answer_not_factually_graded": 11, "wrongful_refusal": 1, "correct_refusal": 3}
- balanced: {"cited_answer_not_factually_graded": 11, "wrongful_refusal": 1, "correct_refusal": 3}
- reranked: {"cited_answer_not_factually_graded": 11, "wrongful_refusal": 1, "correct_refusal": 3}

Citation provenance does not establish that all factual claims are correct.

## Slow queued stand-in trace

| Request ID | Queue ms | Work pre-first-content ms | Work post-first-content ms | Retries | Total ms |
| --- | --- | --- | --- | --- | --- |
| 7e558f9d-4f34-4f7a-a749-bce4f5d6bed9 | 958.15 | 3320.47 | 239.11 | 0 | 4517.73 |

Work includes retrieval, HTTP overhead and model generation. More granular provider phase attribution was not recorded.

## Harmful retry experiment (stand-in)

| Policy | Attempts | Retries | Latency ms | Status |
| --- | --- | --- | --- | --- |
| retry_no_retry | 1 | 0 | 4.85 | rate_limited |
| retry_naive | 3 | 2 | 761.68 | rate_limited |
| retry_deadline_aware | 1 | 0 | 1.6 | rate_limited |

The stand-in laboratory TPM of 50,000 differs from the observed real Groq TPM of 8,000.

The controlled RPM experiment, not the earlier confounded burst run, isolates RPM.

This audit made no API calls and did not modify the original results.csv.
