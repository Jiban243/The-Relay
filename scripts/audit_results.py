
"""Rebuild The Relay experiment tables from local results.

Run:
    python -m scripts.audit_results

No Groq or stand-in API calls are made.
"""

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results"

# Select exact historical runs to avoid mixing
# standalone tests, repeated runs and pilot requests.
RUNS = {
    "steady": ("steady_7b384c6aaa", 8),
    "burst": ("burst_48688426ec", 24),
    "sustained_overload": (
        "sustained_overload_44eea44af7", 48
    ),
    "rpm_halved_initial": (
        "rpm_halved_e0cf5689a4", 24
    ),
    "daily_exhausted": (
        "daily_exhausted_32e13f1639", 12
    ),
    "rpm_control_30": (
        "rpm_control_30_1950bae657", 24
    ),
    "rpm_halved_15": (
        "rpm_halved_15_7f47b2a4c8", 24
    ),
    "quality_full": (
        "quality_full_cca5739647", 45
    ),
    "retry_experiment": (
        "retry_cf4849ebfc", 3
    ),
}

REQUIRED = (
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
)


def number(row, column):
    try:
        value = row.get(column)

        if value in (None, ""):
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def stats(rows, field):
    values = [
        number(row, field)
        for row in rows
    ]

    values = [
        value
        for value in values
        if value is not None
    ]

    return {
        f"p{p}": (
            round(
                float(np.percentile(values, p)),
                2,
            )
            if values else None
        )
        for p in (50, 95, 99)
    }


def sum_known(rows, field):
    values = [
        number(row, field)
        for row in rows
    ]

    valid = [
        value
        for value in values
        if value is not None
    ]

    return {
        "sum": int(sum(valid)) if valid else None,
        "missing": len(rows) - len(valid),
    }


def csv_rows():
    path = DIR / "results.csv"

    with path.open(
        newline="",
        encoding="utf-8",
    ) as stream:

        reader = csv.DictReader(stream)

        absent = (
            set(REQUIRED)
            - set(reader.fieldnames or [])
        )

        if absent:
            raise RuntimeError(
                f"Missing mandatory columns: "
                f"{sorted(absent)}"
            )

        return list(reader)


def select(all_rows, name):
    run_id, expected = RUNS[name]

    picked = [
        row
        for row in all_rows
        if row.get("run_id") == run_id
    ]

    if len(picked) != expected:
        raise RuntimeError(
            f"{name}: expected {expected} records, "
            f"found {len(picked)}"
        )

    ids = {
        row["request_id"]
        for row in picked
    }

    if len(ids) != expected:
        raise RuntimeError(
            f"{name}: duplicate request IDs"
        )

    return picked


def arrival_rate(rows):
    times = sorted(
        datetime.fromisoformat(row["start_time"])
        for row in rows
    )

    gap = (
        times[-1] - times[0]
    ).total_seconds()

    if gap <= 0:
        return None

    return round(
        60 * (len(rows) - 1) / gap,
        2,
    )


def load_summary(rows):
    successful = [
        row
        for row in rows
        if row["status"] == "ok"
    ]

    return {
        "n": len(rows),
        "ok": len(successful),

        "statuses": dict(
            Counter(
                row["status"]
                for row in rows
            )
        ),

        "admissions": dict(
            Counter(
                row.get("admission_reason")
                for row in rows
            )
        ),

        "arrival_rpm": arrival_rate(rows),

        "peak_queue": max(
            int(number(row, "peak_queue") or 0)
            for row in rows
        ),

        "peak_in_flight": max(
            int(number(row, "peak_in_flight") or 0)
            for row in rows
        ),

        "model_first_content_ms": stats(
            successful,
            "first_token_ms",
        ),

        "total_ms": stats(
            successful,
            "total_ms",
        ),

        "tokens_in": sum_known(
            rows, "tokens_in"
        ),

        "tokens_out": sum_known(
            rows, "tokens_out"
        ),

        "retries": sum_known(
            rows, "retries"
        ),
    }


def quality_summary(csv_group, detail):
    expected = {
        (
            item["question_id"],
            item["retrieval_setting"],
        ): item
        for item in detail
    }

    if len(expected) != 45 or len(detail) != 45:
        raise RuntimeError(
            "Expected 45 unique quality JSONL entries"
        )

    observed = {
        (
            row["question_id"],
            row["retrieval_setting"],
        ): row
        for row in csv_group
    }

    if (
        len(observed) != 45
        or observed.keys() != expected.keys()
    ):
        raise RuntimeError(
            "Quality CSV and JSONL do not match"
        )

    for key in expected:

        if (
            expected[key]["trace"]["request_id"]
            != observed[key]["request_id"]
        ):
            raise RuntimeError(
                f"Quality request ID mismatch: {key}"
            )

    report = {}

    for setting in (
        "small",
        "balanced",
        "reranked",
    ):

        group = [
            row
            for row in csv_group
            if row["retrieval_setting"] == setting
        ]

        grades = Counter(
            row["answer_grade"]
            for row in group
            if row.get("answer_grade")
        )

        evidence = Counter()

        for row in group:

            item = expected[
                (
                    row["question_id"],
                    setting,
                )
            ]

            verdict = item["trace"]["verification"]

            supported = item["expected_supported"]

            if verdict in (
                "model_refusal_ungraded",
                "citation_rejected_refused",
                "refused_no_evidence",
            ):

                category = (
                    "wrongful_refusal"
                    if supported
                    else "correct_refusal"
                )

                evidence[category] += 1

            elif verdict == "citation_format_checked":

                # A citation check is not a
                # claim-level factuality check.
                evidence[
                    "cited_answer_not_factually_graded"
                ] += 1

            else:

                evidence["other_unverified"] += 1

        tokens = [
            number(row, "tokens_in")
            + number(row, "tokens_out")
            for row in group
            if (
                number(row, "tokens_in") is not None
                and number(row, "tokens_out") is not None
            )
        ]

        report[setting] = {
            "n": len(group),
            "saved_grades": dict(grades),
            "graded": sum(grades.values()),
            "evidence_outcomes": dict(evidence),

            "tokens_per_answer": (
                round(float(np.mean(tokens)), 2)
                if tokens else None
            ),

            "provider_first_content_ms": stats(
                group,
                "provider_first_content_ms",
            ),

            "buffered_ready_ms": stats(
                group,
                "first_token_ms",
            ),

            "total_ms": stats(
                group,
                "total_ms",
            ),

            "tokens_in": sum_known(
                group, "tokens_in"
            ),

            "tokens_out": sum_known(
                group, "tokens_out"
            ),
        }

        if sum(grades.values()) == 15:

            report[setting][
                "verified_grade_percentages"
            ] = {
                grade: round(
                    100 * grades[grade] / 15,
                    2,
                )
                for grade in (
                    "correct",
                    "wrong",
                    "refused_correctly",
                    "refused_wrongly",
                )
            }

    return report


def slow_trace(groups):
    pool = [
        row
        for group in groups
        for row in group
        if (
            row["status"] == "ok"
            and (number(row, "queue_wait_ms") or 0) > 0
            and number(row, "first_token_ms") is not None
        )
    ]

    if not pool:
        return None

    row = max(
        pool,
        key=lambda item: (
            number(item, "total_ms") or 0
        ),
    )

    queue = number(row, "queue_wait_ms")
    first = number(row, "first_token_ms")
    total = number(row, "total_ms")

    return {
        "request_id": row["request_id"],
        "scenario": row["scenario"],

        "queue_ms": queue,

        "working_before_first_model_content_ms": (
            round(max(0, first - queue), 2)
        ),

        "working_after_first_model_content_ms": (
            round(max(0, total - first), 2)
        ),

        "retries": int(
            number(row, "retries") or 0
        ),

        "first_model_content_ms": first,

        "buffered_ready_ms": number(
            row,
            "user_visible_first_content_ms",
        ),

        "total_ms": total,
    }


def md_table(headers, records):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(
            ["---"] * len(headers)
        ) + " |",
    ]

    lines += [
        "| " + " | ".join(
            str(value) for value in row
        ) + " |"
        for row in records
    ]

    return "\n".join(lines)


def create_markdown(report):
    lines = [
        "# The Relay — Results Audit",
        "",
        "Recomputed from selected run IDs in results.csv.",
        "",
        "## Five stand-in load scenarios",
        "",
    ]

    records = []

    names = (
        "steady",
        "burst",
        "sustained_overload",
        "rpm_halved_initial",
        "daily_exhausted",
    )

    for name in names:

        item = report["load"][name]

        records.append([
            name,
            item["n"],
            item["ok"],
            item["statuses"].get("timeout", 0),
            item["statuses"].get("rate_limited", 0),
            item["statuses"].get("error", 0),
            item["peak_queue"],
            item["model_first_content_ms"]["p50"],
            item["model_first_content_ms"]["p95"],
            item["model_first_content_ms"]["p99"],
            item["total_ms"]["p50"],
            item["total_ms"]["p95"],
            item["total_ms"]["p99"],
        ])

    lines += [
        md_table(
            [
                "Scenario", "N", "OK", "Timeout",
                "Rate limited", "Error", "Peak queue",
                "Model TTFT p50", "p95", "p99",
                "Total p50", "p95", "p99",
            ],
            records,
        ),
        "",
        "Percentiles use successful responses only.",
        "Stand-in tokens and latency are simulated.",
        "The application buffers its response, so "
        "model TTFT is not user-visible TTFT.",
        "",
        "## Controlled simulated RPM experiment",
        "",
    ]

    lines += [
        md_table(
            [
                "Run",
                "N",
                "OK",
                "Queue timed out",
                "Arrival RPM",
            ],
            [
                [
                    name,
                    report["load"][name]["n"],
                    report["load"][name]["ok"],
                    report["load"][name]["admissions"].get(
                        "queue_timeout", 0
                    ),
                    report["load"][name]["arrival_rpm"],
                ]
                for name in (
                    "rpm_control_30",
                    "rpm_halved_15",
                )
            ],
        ),
        "",
        "## Real Groq retrieval evaluation",
        "",
    ]

    lines += [
        md_table(
            [
                "Setting",
                "N",
                "Mean tokens/answer",
                "Provider first-content p50",
                "p95",
                "Total p50",
                "p95",
                "Saved grades",
            ],
            [
                [
                    name,
                    item["n"],
                    item["tokens_per_answer"],
                    item["provider_first_content_ms"]["p50"],
                    item["provider_first_content_ms"]["p95"],
                    item["total_ms"]["p50"],
                    item["total_ms"]["p95"],
                    f"{item['graded']}/15",
                ]
                for name, item in report["quality"].items()
            ],
        ),
        "",
        "Evidence screening (not independent factual grading):",
        "",
    ]

    for name, item in report["quality"].items():

        lines.append(
            f"- {name}: "
            f"{json.dumps(item['evidence_outcomes'])}"
        )

    lines += [
        "",
        "Citation provenance does not establish that "
        "all factual claims are correct.",
        "",
        "## Slow queued stand-in trace",
        "",
    ]

    trace = report["slow_trace"]

    if trace:

        lines += [
            md_table(
                [
                    "Request ID",
                    "Queue ms",
                    "Work pre-first-content ms",
                    "Work post-first-content ms",
                    "Retries",
                    "Total ms",
                ],
                [[
                    trace["request_id"],
                    trace["queue_ms"],
                    trace[
                        "working_before_first_model_content_ms"
                    ],
                    trace[
                        "working_after_first_model_content_ms"
                    ],
                    trace["retries"],
                    trace["total_ms"],
                ]],
            ),
            "",
            "Work includes retrieval, HTTP overhead and "
            "model generation. More granular provider "
            "phase attribution was not recorded.",
        ]

    else:

        lines.append(
            "No queued successful request with a "
            "first-content measurement exists."
        )

    lines += [
        "",
        "## Harmful retry experiment (stand-in)",
        "",
        md_table(
            [
                "Policy",
                "Attempts",
                "Retries",
                "Latency ms",
                "Status",
            ],
            [
                [
                    name,
                    item["retries"] + 1,
                    item["retries"],
                    item["total_ms"],
                    item["status"],
                ]
                for name, item in report["retry"].items()
            ],
        ),
        "",
        "The stand-in laboratory TPM of 50,000 differs "
        "from the observed real Groq TPM of 8,000.",
        "",
        "The controlled RPM experiment, not the "
        "earlier confounded burst run, isolates RPM.",
        "",
        "This audit made no API calls and did not "
        "modify the original results.csv.",
        "",
    ]

    return "\n".join(lines)


def main():
    rows = csv_rows()

    selected = {
        name: select(rows, name)
        for name in RUNS
    }

    detail_file = (
        DIR / f"{RUNS['quality_full'][0]}.jsonl"
    )

    with detail_file.open(
        encoding="utf-8"
    ) as file:

        details = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    retry = {}

    for row in selected["retry_experiment"]:

        retry[row["scenario"]] = {
            "status": row["status"],
            "retries": int(
                number(row, "retries") or 0
            ),
            "total_ms": number(
                row, "total_ms"
            ),
        }

    required_policies = {
        "retry_no_retry",
        "retry_naive",
        "retry_deadline_aware",
    }

    if set(retry) != required_policies:
        raise RuntimeError(
            f"Unexpected retry policies: "
            f"{set(retry)}"
        )

    names = (
        "steady",
        "burst",
        "sustained_overload",
        "rpm_halved_initial",
        "daily_exhausted",
    )

    report = {
        "load": {
            name: load_summary(selected[name])
            for name in RUNS
            if name not in (
                "quality_full",
                "retry_experiment",
            )
        },

        "quality": quality_summary(
            selected["quality_full"],
            details,
        ),

        "retry": retry,

        "slow_trace": slow_trace(
            [
                selected[name]
                for name in names
            ]
        ),
    }

    output_json = (
        DIR / "FINAL_MEASUREMENT_AUDIT.json"
    )

    output_md = (
        DIR / "FINAL_MEASUREMENT_AUDIT.md"
    )

    output_json.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    output_md.write_text(
        create_markdown(report),
        encoding="utf-8",
    )

    print("\n========== RESULTS AUDIT ==========")

    print("Source: Saved experiment results")
    print("No API calls made.\n")

    for name, item in report["load"].items():

        print(
            f"{name}: "
            f"{item['ok']}/{item['n']} successful; "
            f"peak queue {item['peak_queue']}"
        )

    print("\nQuality evaluation:")

    for name, item in report["quality"].items():

        print(
            f"{name}: {item['graded']}/15 graded; "
            f"evidence outcomes "
            f"{item['evidence_outcomes']}"
        )

    print("\nSlow trace:")

    print(
        json.dumps(
            report["slow_trace"],
            indent=2,
        )
    )

    print("\nSaved:", output_md)
    print("Saved:", output_json)

    print(
        "\nOriginal results.csv was not modified."
    )


if __name__ == "__main__":
    main()
