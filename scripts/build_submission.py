
"""Build The Relay submission write-up from saved results.

Run:
    python -m scripts.build_submission

No API calls. Does not modify results.csv.
"""

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "SUBMISSION_WRITEUP.md"

RUNS = {
    "Steady": "steady_7b384c6aaa",
    "Burst": "burst_48688426ec",
    "Sustained overload": "sustained_overload_44eea44af7",
    "RPM halved (initial)": "rpm_halved_e0cf5689a4",
    "Daily budget exhausted": "daily_exhausted_32e13f1639",
}

CONTROL = {
    "30 RPM": "rpm_control_30_1950bae657",
    "15 RPM": "rpm_halved_15_7f47b2a4c8",
}

QUALITY_RUN = "quality_full_cca5739647"
RETRY_RUN = "retry_cf4849ebfc"


def number(row, field):
    value = row.get(field)

    if value in ("", None):
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def fmt(value, digits=1):
    if value is None:
        return "not measured"

    return f"{value:.{digits}f}"


def percent(count, total):
    if not total:
        return "not measured"

    return f"{100 * count / total:.1f}%"


def average(values):
    if not values:
        return None

    return float(np.mean(values))


def percentiles(rows, field, points=(50, 95, 99)):
    values = [
        number(row, field)
        for row in rows
    ]

    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return "not measured"

    return " / ".join(
        fmt(float(np.percentile(values, p)))
        for p in points
    )


def offered_rate(rows):
    if len(rows) < 2:
        return None

    times = sorted(
        datetime.fromisoformat(row["start_time"])
        for row in rows
    )

    duration = (
        times[-1] - times[0]
    ).total_seconds()

    if duration <= 0:
        return None

    return 60 * (len(rows) - 1) / duration


def markdown_table(headers, rows):
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(
            "---" for _ in headers
        ) + " |",
    ]

    for row in rows:
        output.append(
            "| "
            + " | ".join(str(value) for value in row)
            + " |"
        )

    return "\n".join(output)


def main():
    csv_path = RESULTS / "results.csv"

    with csv_path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        all_rows = list(csv.DictReader(file))

    by_run = {}

    for row in all_rows:
        by_run.setdefault(
            row.get("run_id"),
            [],
        ).append(row)

    expected_counts = {
        "steady_7b384c6aaa": 8,
        "burst_48688426ec": 24,
        "sustained_overload_44eea44af7": 48,
        "rpm_halved_e0cf5689a4": 24,
        "daily_exhausted_32e13f1639": 12,
        "rpm_control_30_1950bae657": 24,
        "rpm_halved_15_7f47b2a4c8": 24,
        QUALITY_RUN: 45,
        RETRY_RUN: 3,
    }

    for run_id, count in expected_counts.items():
        rows = by_run.get(run_id, [])

        if len(rows) != count:
            raise ValueError(
                f"{run_id}: expected {count} rows, "
                f"found {len(rows)}"
            )

        if len({
            row["request_id"] for row in rows
        }) != count:
            raise ValueError(
                f"Duplicate request IDs: {run_id}"
            )

    # ------------------------------------------
    # TABLE 2A: LOAD LATENCY
    # ------------------------------------------

    table_2a = []
    table_2b = []

    for scenario, run_id in RUNS.items():
        rows = by_run[run_id]

        successful = [
            row for row in rows
            if row["status"] == "ok"
        ]

        statuses = Counter(
            row["status"] for row in rows
        )

        durations = [
            number(row, "duration_s")
            for row in rows
        ]

        durations = [
            item for item in durations
            if item is not None
        ]

        table_2a.append([
            scenario,
            "stand-in",
            len(rows),
            fmt(durations[0] if durations else None),
            fmt(offered_rate(rows)),
            percentiles(
                successful,
                "first_token_ms",
            ),
            percentiles(
                successful,
                "total_ms",
            ),
        ])

        token_totals = []

        for row in successful:
            tokens_in = number(
                row, "tokens_in"
            )
            tokens_out = number(
                row, "tokens_out"
            )

            if (
                tokens_in is not None
                and tokens_out is not None
            ):
                token_totals.append(
                    tokens_in + tokens_out
                )

        retries = [
            number(row, "retries")
            for row in rows
        ]

        retries = [
            value for value in retries
            if value is not None
        ]

        peaks = [
            number(row, "peak_queue")
            for row in rows
        ]

        peaks = [
            value for value in peaks
            if value is not None
        ]

        retry_average = (
            sum(retries) / len(rows)
            if len(retries) == len(rows)
            else None
        )

        table_2b.append([
            scenario,
            percent(
                statuses["ok"],
                len(rows),
            ),
            percent(
                statuses["rate_limited"],
                len(rows),
            ),
            percent(
                statuses["timeout"] + statuses["error"],
                len(rows),
            ),
            fmt(retry_average, 2),
            fmt(max(peaks) if peaks else None, 0),
            fmt(average(token_totals)),
            "not measured",
        ])

    # ------------------------------------------
    # TABLE 3: QUALITY AND RETRIEVAL
    # ------------------------------------------

    quality_rows = by_run[QUALITY_RUN]
    table_3 = []

    configurations = [
        ("small", "250 words / 2 / no"),
        ("balanced", "500 words / 4 / no"),
        ("reranked", "500 words / 4 / lexical"),
    ]

    valid_grades = {
        "correct",
        "wrong",
        "refused_correctly",
        "refused_wrongly",
    }

    for setting, configuration in configurations:
        rows = [
            row for row in quality_rows
            if row["retrieval_setting"] == setting
        ]

        if len(rows) != 15:
            raise ValueError(
                f"Expected 15 records for {setting}"
            )

        if any(
            row["answer_grade"] not in valid_grades
            for row in rows
        ):
            raise ValueError(
                f"Incomplete quality grades: {setting}"
            )

        counts = Counter(
            row["answer_grade"]
            for row in rows
        )

        token_totals = []

        for row in rows:
            tokens_in = number(
                row, "tokens_in"
            )
            tokens_out = number(
                row, "tokens_out"
            )

            if (
                tokens_in is not None
                and tokens_out is not None
            ):
                token_totals.append(
                    tokens_in + tokens_out
                )

        table_3.append([
            f"{setting} ({configuration})",
            len(rows),
            percent(counts["correct"], 15),
            percent(
                counts["wrong"] + counts["refused_wrongly"],
                15,
            ),
            percent(
                counts["refused_correctly"],
                15,
            ),
            fmt(average(token_totals)),
            percentiles(
                rows,
                "provider_first_content_ms",
                (50, 95),
            ),
            percentiles(
                rows,
                "total_ms",
                (50, 95),
            ),
        ])

    # ------------------------------------------
    # CONTROLLED RPM COMPARISON
    # ------------------------------------------

    baseline = by_run[CONTROL["30 RPM"]]
    reduced = by_run[CONTROL["15 RPM"]]

    baseline_success = sum(
        row["status"] == "ok"
        for row in baseline
    )

    reduced_success = sum(
        row["status"] == "ok"
        for row in reduced
    )

    success_difference = (
        100
        * (baseline_success - reduced_success)
        / 24
    )

    # ------------------------------------------
    # RETRY AND SLOW TRACE
    # ------------------------------------------

    retry_rows = {
        row["scenario"]: row
        for row in by_run[RETRY_RUN]
    }

    required_policies = {
        "retry_no_retry",
        "retry_naive",
        "retry_deadline_aware",
    }

    if set(retry_rows) != required_policies:
        raise ValueError(
            "Missing retry experiment records"
        )

    no_retry = number(
        retry_rows["retry_no_retry"],
        "total_ms",
    )

    naive_retry = number(
        retry_rows["retry_naive"],
        "total_ms",
    )

    deadline_aware = number(
        retry_rows["retry_deadline_aware"],
        "total_ms",
    )

    slow = json.loads(
        (
            RESULTS / "FINAL_MEASUREMENT_AUDIT.json"
        ).read_text(encoding="utf-8")
    )["slow_trace"]

    if not slow:
        raise ValueError(
            "No recorded slow-request trace"
        )

    # ------------------------------------------
    # REQUIRED WRITE-UP HEADINGS AND TABLES
    # ------------------------------------------

    sections = [
        "# The Relay — Submission Write-up",

        (
            "*All table values are recalculated from "
            "selected run IDs in `results.csv`. "
            "Historical runs remain unchanged.*"
        ),

        "## 1. What I built and why",

        (
            "A Python research assistant restricted to three "
            "exact arXiv paper versions. Its five stages are "
            "question routing, BM25 passage retrieval, "
            "grounded prompt construction, streaming Groq "
            "generation, and citation validation. It can "
            "revise a citation-invalid response once. "
            "Explicit paper routing is central: without "
            "using it during retrieval, the Transformer "
            "architecture question selected H-Net evidence "
            "and produced a wrongful refusal. I fixed "
            "routing but preserved the original benchmark."
        ),

        "## 2. The real service and my stand-in",

        (
            "Real service: Groq `openai/gpt-oss-20b` with "
            "streaming and low reasoning effort. "
            "Three successful calibration calls measured "
            "first-content p50=884.57 ms and total "
            "p50=983.85 ms (n=3, insufficient for reliable "
            "tail estimates). An earlier 12-call calibration "
            "produced truncated completions, not provider "
            "outages. Response headers showed 8,000 TPM "
            "and 1,000 RPD; other limits came from "
            "documentation. No real HTTP 429 was reached. "
            "The FastAPI stand-in samples successful "
            "latency pairs and simulates SSE streaming, "
            "quotas, overload slowdown, errors and tails. "
            "Its token estimates, failure distribution "
            "and 50,000 laboratory TPM are synthetic."
        ),

        "**Table 1 — Real-service limits**",

        markdown_table(
            [
                "Limit",
                "Value",
                "How established",
                "Confidence",
            ],
            [
                [
                    "RPM", "30",
                    "Groq docs; not reached", "medium",
                ],
                [
                    "TPM", "8,000",
                    "Response header and docs", "high",
                ],
                [
                    "RPD", "1,000",
                    "Response header and docs", "high",
                ],
                [
                    "TPD", "200,000",
                    "Groq docs only", "medium",
                ],
                [
                    "Concurrent calls",
                    "not measured",
                    "No saturation test on Groq",
                    "low",
                ],
                [
                    "Limit response",
                    "HTTP 429",
                    "Documented, not observed",
                    "medium",
                ],
                [
                    "Recovery time",
                    "not measured",
                    "Reset hints seen on HTTP 200",
                    "low",
                ],
            ],
        ),

        "## 3. Results",

        "**Table 2a — Latency (milliseconds)**",

        markdown_table(
            [
                "Scenario",
                "Source",
                "Requests",
                "Duration (s)",
                "Offered /min",
                "First content p50/p95/p99",
                "Total p50/p95/p99",
            ],
            table_2a,
        ),

        "**Table 2b — Outcomes**",

        markdown_table(
            [
                "Scenario",
                "Succeeded",
                "Rate-limited",
                "Timed out / gave up",
                "Retries / req",
                "Peak queue",
                "Tokens / answer",
                "Cost / 1,000",
            ],
            table_2b,
        ),

        (
            "All five load runs use the stand-in. "
            "Latency percentiles include successful requests "
            "only. `first_token_ms` measures first *model* "
            "content after admission; buffered user-visible "
            "response readiness is separately recorded in "
            "`user_visible_first_content_ms`. Queue-full "
            "errors count as given up. Costs are not "
            "measured for synthetic calls."
        ),

        (
            f"A controlled 24-request comparison, changing "
            f"only 30 to 15 RPM, reduced successes from "
            f"{baseline_success}/24 to {reduced_success}/24 "
            f"({success_difference:.1f} percentage points). "
            f"The measured offered rate was "
            f"{fmt(offered_rate(baseline))}/min over a short "
            f"window, not sustained minute-long throughput. "
            f"Separately, 45 real Groq quality-evaluation "
            f"calls completed without HTTP errors."
        ),

        "## 4. Where the time goes",

        (
            f"Slow stand-in request `{slow['request_id']}`: "
            f"queue {fmt(slow['queue_ms'], 2)} ms; "
            f"working before first model content "
            f"{fmt(slow['working_before_first_model_content_ms'], 2)} ms; "
            f"working afterward "
            f"{fmt(slow['working_after_first_model_content_ms'], 2)} ms; "
            f"retries {slow['retries']}; "
            f"total {fmt(slow['total_ms'], 2)} ms. "
            f"First model content arrived at "
            f"{fmt(slow['first_model_content_ms'], 2)} ms; "
            f"buffered response was ready at "
            f"{fmt(slow['buffered_ready_ms'], 2)} ms. "
            f"Working includes application and network "
            f"overhead; provider computation was not "
            f"measured separately."
        ),

        "## 5. When a limit is hit",

        (
            "The application uses bounded FIFO queueing "
            "with request/token reservations, concurrency "
            "caps and wait deadlines. Per-minute exhaustion "
            "can delay callers and end in `queue_timeout`; "
            "queue-full requests are dropped. Exhausting "
            "the daily quota causes immediate rejection "
            "without a guessed answer. A deadline-aware "
            "retry module was tested separately but is "
            "not connected to normal pipeline calls."
        ),

        "## Retrieval: quality versus cost",

        "**Table 3 — Real Groq retrieval evaluation**",

        markdown_table(
            [
                "Setting (words / top-k / rerank)",
                "Questions",
                "Correct and cited",
                "Wrong",
                "Correctly refused",
                "Tokens / answer",
                "First content p50/p95",
                "Total p50/p95",
            ],
            table_3,
        ),

        (
            "These are rubric-screened grades, not "
            "independent expert verification: six existing "
            "human decisions were preserved and 39 answers "
            "were graded using an offline concept/citation "
            "rubric. Every setting had one wrongful "
            "Transformer refusal and three appropriate "
            "out-of-library refusals. Balanced retrieval "
            "is my default because of its less extreme "
            "observed tail latency; I would manually use "
            "small retrieval when conserving tokens. "
            "Automatic retrieval switching is not built."
        ),

        "## How this library will mislead a naive retriever",

        (
            "1. **Cross-paper terms:** “What are the "
            "encoder and decoder components of the "
            "Transformer?” retrieved H-Net passages and "
            "caused wrongful refusals. "
            "2. **PDF extraction noise:** Questions about "
            "H-Net's chunk boundaries retrieve garbled "
            "figure/equation text, confusing ranking. "
            "3. **Incidental word overlap:** A stock-price "
            "question retrieves unrelated Moshi pages; "
            "the correct response is a refusal. "
            "Duplicate chunks from one Moshi page also "
            "reduced evidence diversity."
        ),

        "## 6. Retries",

        (
            f"The policy respects `Retry-After`, skips "
            f"daily quota retries and partial-stream replay, "
            f"and only retries transient errors within a "
            f"deadline. In an exhausted simulated 1-RPM "
            f"window, no retry took {fmt(no_retry, 2)} ms; "
            f"naive retries took {fmt(naive_retry, 2)} ms, "
            f"adding {fmt(naive_retry - no_retry, 2)} ms "
            f"and two failed provider attempts. "
            f"Deadline-aware retry stopped after one "
            f"attempt ({fmt(deadline_aware, 2)} ms)."
        ),

        "## 7. What the person sees",

        (
            "> Today's research-answer budget is exhausted. "
            "I cannot verify an answer now and will not "
            "guess. Please try after the quota resets."
        ),

        (
            "This is the proposed 3pm user-facing message. "
            "The experimental runner records rejections "
            "but does not implement a customer-facing UI."
        ),

        (
            "## 8. What I'm unsure about, "
            "what I left out, what I learned"
        ),

        (
            "The weakest assumption is that three real "
            "latency samples represent service behavior "
            "under concurrency. If not, the simulated "
            "tail results are poor predictors of Groq. "
            "I omitted a vector database and a production "
            "UI to prioritize rate limits and traceability. "
            "The key lesson: retries can spend the caller's "
            "entire patience budget without restoring "
            "capacity."
        ),

        "## Your feedback on Vaani",

        (
            "Website-only impression, not a product test: "
            "the emphasis on natural turn-taking, language "
            "switching and reducing dead air stood out. "
            "The homepage's roughly 450-ms p90 latency "
            "claim and the developers page's 900–1,200-ms "
            "full-turn p90 could be easier to distinguish. "
            "I would place consistent measurement definitions "
            "beside each performance claim."
        ),

        "## How I worked with the assistant",

        (
            "The real conversation history is attached "
            "separately, without cleaning or rewriting it. "
            "The three most useful prompts concerned: "
            "(1) restricting answers to three exact papers "
            "and requiring citations, "
            "(2) demanding complete copy-paste-ready Python "
            "files with local tests, and "
            "(3) requiring the assistant to assess answers "
            "rather than delegate 45 manual grading tasks. "
            "A mistake I caught was rejection of valid "
            "Unicode-bracket citations; examining the saved "
            "model output exposed the bug. I overruled "
            "manual per-answer grading and used an auditable "
            "offline rubric instead, explicitly acknowledging "
            "its limits. My concrete work includes running "
            "the experiments, verifying the outputs, "
            "challenging failures and preserving the "
            "original benchmark. The code was AI-assisted."
        ),

        (
            "**Reproducibility:** `python -m pytest tests/ -q`, "
            "`python -m scripts.audit_results`, "
            "`python -m scripts.build_submission`. "
            "Raw data: `results/results.csv`, "
            "`results/quality_full_cca5739647.jsonl`, "
            "`results/QUALITY_GRADE_AUDIT.jsonl` and "
            "`results/FINAL_MEASUREMENT_AUDIT.json`."
        ),

        (
            "**References:** "
            "https://console.groq.com/docs/rate-limits ; "
            "https://console.groq.com/docs/models ; "
            "https://www.vaani.dev/ ; "
            "https://www.vaani.dev/developers"
        ),
    ]

    OUTPUT.write_text(
        "\n\n".join(sections) + "\n",
        encoding="utf-8",
    )

    print("\n========== SUBMISSION WRITE-UP ==========")
    print("Created:", OUTPUT)
    print("Five load scenarios:", len(table_2a))
    print("Retrieval settings:", len(table_3))
    print("Quality run:", QUALITY_RUN)
    print("Raw results.csv: unchanged")
    print()
    print(
        "Review first-person website feedback and "
        "contribution statements for accuracy."
    )
    print(
        "Check pagination before submission: "
        "the assignment allows two pages of writing."
    )


if __name__ == "__main__":
    main()
