# The Relay — Submission Write-up

*All table values are recalculated from selected run IDs in `results.csv`. Historical runs remain unchanged.*

## 1. What I built and why

A Python research assistant restricted to three exact arXiv paper versions. Its five stages are question routing, BM25 passage retrieval, grounded prompt construction, streaming Groq generation, and citation validation. It can revise a citation-invalid response once. Explicit paper routing is central: without using it during retrieval, the Transformer architecture question selected H-Net evidence and produced a wrongful refusal. I fixed routing but preserved the original benchmark.

## 2. The real service and my stand-in

Real service: Groq `openai/gpt-oss-20b` with streaming and low reasoning effort. Three successful calibration calls measured first-content p50=884.57 ms and total p50=983.85 ms (n=3, insufficient for reliable tail estimates). An earlier 12-call calibration produced truncated completions, not provider outages. Response headers showed 8,000 TPM and 1,000 RPD; other limits came from documentation. No real HTTP 429 was reached. The FastAPI stand-in samples successful latency pairs and simulates SSE streaming, quotas, overload slowdown, errors and tails. Its token estimates, failure distribution and 50,000 laboratory TPM are synthetic.

**Table 1 — Real-service limits**

| Limit | Value | How established | Confidence |
| --- | --- | --- | --- |
| RPM | 30 | Groq docs; not reached | medium |
| TPM | 8,000 | Response header and docs | high |
| RPD | 1,000 | Response header and docs | high |
| TPD | 200,000 | Groq docs only | medium |
| Concurrent calls | not measured | No saturation test on Groq | low |
| Limit response | HTTP 429 | Documented, not observed | medium |
| Recovery time | not measured | Reset hints seen on HTTP 200 | low |

## 3. Results

**Table 2a — Latency (milliseconds)**

| Scenario | Source | Requests | Duration (s) | Offered /min | First content p50/p95/p99 | Total p50/p95/p99 |
| --- | --- | --- | --- | --- | --- | --- |
| Steady | stand-in | 8 | 18.0 | 24.0 | 904.3 / 961.0 / 969.4 | 998.0 / 1048.7 / 1050.1 |
| Burst | stand-in | 24 | 2.7 | 5971.4 | 1664.6 / 2452.0 / 2513.6 | 1760.5 / 2602.7 / 2678.9 |
| Sustained overload | stand-in | 48 | 8.3 | 399.9 | 1670.8 / 4167.0 / 4256.3 | 1780.0 / 4408.1 / 4495.8 |
| RPM halved (initial) | stand-in | 24 | 3.6 | 999.2 | 1928.5 / 2566.6 / 2672.0 | 2010.8 / 2731.0 / 2833.6 |
| Daily budget exhausted | stand-in | 12 | 1.6 | 665.9 | 1162.0 / 1301.6 / 1314.1 | 1249.4 / 1398.3 / 1411.5 |

**Table 2b — Outcomes**

| Scenario | Succeeded | Rate-limited | Timed out / gave up | Retries / req | Peak queue | Tokens / answer | Cost / 1,000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Steady | 100.0% | 0.0% | 0.0% | 0.00 | 0 | 1030.1 | not measured |
| Burst | 25.0% | 0.0% | 75.0% | 0.00 | 12 | 1054.5 | not measured |
| Sustained overload | 20.8% | 0.0% | 79.2% | 0.00 | 7 | 1127.5 | not measured |
| RPM halved (initial) | 37.5% | 0.0% | 62.5% | 0.00 | 12 | 952.2 | not measured |
| Daily budget exhausted | 25.0% | 75.0% | 0.0% | 0.00 | 0 | 1058.3 | not measured |

All five load runs use the stand-in. Latency percentiles include successful requests only. `first_token_ms` measures first *model* content after admission; buffered user-visible response readiness is separately recorded in `user_visible_first_content_ms`. Queue-full errors count as given up. Costs are not measured for synthetic calls.

A controlled 24-request comparison, changing only 30 to 15 RPM, reduced successes from 24/24 to 15/24 (37.5 percentage points). The measured offered rate was 133.3/min over a short window, not sustained minute-long throughput. Separately, 45 real Groq quality-evaluation calls completed without HTTP errors.

## 4. Where the time goes

Slow stand-in request `7e558f9d-4f34-4f7a-a749-bce4f5d6bed9`: queue 958.15 ms; working before first model content 3320.47 ms; working afterward 239.11 ms; retries 0; total 4517.73 ms. First model content arrived at 4278.62 ms; buffered response was ready at 4517.70 ms. Working includes application and network overhead; provider computation was not measured separately.

## 5. When a limit is hit

The application uses bounded FIFO queueing with request/token reservations, concurrency caps and wait deadlines. Per-minute exhaustion can delay callers and end in `queue_timeout`; queue-full requests are dropped. Exhausting the daily quota causes immediate rejection without a guessed answer. A deadline-aware retry module was tested separately but is not connected to normal pipeline calls.

## Retrieval: quality versus cost

**Table 3 — Real Groq retrieval evaluation**

| Setting (words / top-k / rerank) | Questions | Correct and cited | Wrong | Correctly refused | Tokens / answer | First content p50/p95 | Total p50/p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small (250 words / 2 / no) | 15 | 73.3% | 6.7% | 20.0% | 1110.9 | 1309.8 / 13617.1 | 1358.7 / 14033.2 |
| balanced (500 words / 4 / no) | 15 | 73.3% | 6.7% | 20.0% | 2879.6 | 1132.5 / 5624.6 | 1547.2 / 6300.1 |
| reranked (500 words / 4 / lexical) | 15 | 73.3% | 6.7% | 20.0% | 3085.3 | 1224.7 / 5161.3 | 1587.8 / 5519.8 |

These are rubric-screened grades, not independent expert verification: six existing human decisions were preserved and 39 answers were graded using an offline concept/citation rubric. Every setting had one wrongful Transformer refusal and three appropriate out-of-library refusals. Balanced retrieval is my default because of its less extreme observed tail latency; I would manually use small retrieval when conserving tokens. Automatic retrieval switching is not built.

## How this library will mislead a naive retriever

1. **Cross-paper terms:** “What are the encoder and decoder components of the Transformer?” retrieved H-Net passages and caused wrongful refusals. 2. **PDF extraction noise:** Questions about H-Net's chunk boundaries retrieve garbled figure/equation text, confusing ranking. 3. **Incidental word overlap:** A stock-price question retrieves unrelated Moshi pages; the correct response is a refusal. Duplicate chunks from one Moshi page also reduced evidence diversity.

## 6. Retries

The policy respects `Retry-After`, skips daily quota retries and partial-stream replay, and only retries transient errors within a deadline. In an exhausted simulated 1-RPM window, no retry took 4.85 ms; naive retries took 761.68 ms, adding 756.83 ms and two failed provider attempts. Deadline-aware retry stopped after one attempt (1.60 ms).

## 7. What the person sees

> Today's research-answer budget is exhausted. I cannot verify an answer now and will not guess. Please try after the quota resets.

This is the proposed 3pm user-facing message. The experimental runner records rejections but does not implement a customer-facing UI.

## 8. What I'm unsure about, what I left out, what I learned

The weakest assumption is that three real latency samples represent service behavior under concurrency. If not, the simulated tail results are poor predictors of Groq. I omitted a vector database and a production UI to prioritize rate limits and traceability. The key lesson: retries can spend the caller's entire patience budget without restoring capacity.

## Your feedback on Vaani

Website-only impression, not a product test: the emphasis on natural turn-taking, language switching and reducing dead air stood out. The homepage's roughly 450-ms p90 latency claim and the developers page's 900–1,200-ms full-turn p90 could be easier to distinguish. I would place consistent measurement definitions beside each performance claim.

## How I worked with the assistant

The real conversation history is attached separately, without cleaning or rewriting it. The three most useful prompts concerned: (1) restricting answers to three exact papers and requiring citations, (2) demanding complete copy-paste-ready Python files with local tests, and (3) requiring the assistant to assess answers rather than delegate 45 manual grading tasks. A mistake I caught was rejection of valid Unicode-bracket citations; examining the saved model output exposed the bug. I overruled manual per-answer grading and used an auditable offline rubric instead, explicitly acknowledging its limits. My concrete work includes running the experiments, verifying the outputs, challenging failures and preserving the original benchmark. The code was AI-assisted.

**Reproducibility:** `python -m pytest tests/ -q`, `python -m scripts.audit_results`, `python -m scripts.build_submission`. Raw data: `results/results.csv`, `results/quality_full_cca5739647.jsonl`, `results/QUALITY_GRADE_AUDIT.jsonl` and `results/FINAL_MEASUREMENT_AUDIT.json`.

**References:** https://console.groq.com/docs/rate-limits ; https://console.groq.com/docs/models ; https://www.vaani.dev/ ; https://www.vaani.dev/developers
