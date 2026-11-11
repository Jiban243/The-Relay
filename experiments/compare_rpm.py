
"""Controlled comparison of 30 RPM versus 15 RPM.

Uses the existing load runner and calibrated stand-in.
Does not call the real Groq API.

Run:
    python -m experiments.compare_rpm
"""

import asyncio
import csv
from collections import Counter
from datetime import datetime

from experiments.run_load import (
    SCENARIOS,
    CSV_PATH,
    run_scenario,
)


# Identical workload for both experiments.
BASE_CONFIG = {
    "count": 24,
    "interval_s": 0.45,
    "rpd": 1000,
    "in_flight": 8,
    "queue": 12,
    "wait_s": 1.0,

    # No injected failures or tail events.
    "fail_prob": 0.0,
    "timeout_prob": 0.0,
    "tail_prob": 0.0,
}


def load_run_rows(run_id):
    """Read only the rows belonging to one run."""

    with CSV_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        return [
            row
            for row in csv.DictReader(file)
            if row["run_id"] == run_id
        ]


def calculate_offered_rate(rows):
    """
    Measured arrival rate for regularly
    scheduled requests.

    Uses the actual first and last arrival times.
    """

    if len(rows) < 2:
        return None

    times = sorted(
        datetime.fromisoformat(row["start_time"])
        for row in rows
    )

    elapsed_s = (
        times[-1] - times[0]
    ).total_seconds()

    if elapsed_s <= 0:
        return None

    return round(
        60 * (len(times) - 1) / elapsed_s,
        2,
    )


def explain_run(summary):
    """Report outcomes and admission decisions."""

    run_id = summary["run_id"]
    rows = load_run_rows(run_id)

    status_counts = Counter(
        row["status"]
        for row in rows
    )

    admission_counts = Counter(
        row["admission_reason"]
        for row in rows
    )

    succeeded = status_counts["ok"]

    success_pct = (
        100 * succeeded / len(rows)
        if rows
        else 0
    )

    offered_rate = calculate_offered_rate(rows)

    sim_metrics = summary["standin_metrics"]

    print("\n" + "=" * 55)
    print("Scenario:", summary["scenario"])
    print("Run ID:", run_id)

    print("\nWorkload")
    print("Requests offered:", len(rows))
    print("Configured RPM:", summary["config"]["rpm"])
    print("Measured offered rate:", offered_rate, "/min")

    print("\nOutcomes")
    print("Succeeded:", succeeded)
    print("Success rate:", round(success_pct, 1), "%")
    print("Status counts:", dict(status_counts))

    print("\nAdmission decisions")
    print(dict(admission_counts))

    print("\nCapacity")
    print(
        "Stand-in admitted requests:",
        sim_metrics["minute_requests"]
    )
    print(
        "Application peak queue:",
        summary["app_peak_queue"]
    )
    print(
        "Application peak in-flight:",
        summary["app_peak_in_flight"]
    )

    print("\nLatency (successful requests)")
    print(
        "First model content:",
        summary["model_ttft_ms"]
    )
    print(
        "Total time:",
        summary["successful_total_ms"]
    )

    return {
        "run_id": run_id,
        "successful": succeeded,
        "success_pct": success_pct,
        "offered_rate": offered_rate,
    }


async def main():

    print("\nTHE RELAY — CONTROLLED RPM COMPARISON")
    print("Source: Stand-in model only")
    print("No real Groq quota is consumed.")

    # Both scenarios inherit exactly the same
    # configuration, except for the RPM limit.
    SCENARIOS["rpm_control_30"] = {
        **BASE_CONFIG,
        "rpm": 30,
    }

    SCENARIOS["rpm_halved_15"] = {
        **BASE_CONFIG,
        "rpm": 15,
    }

    # Each run starts its own fresh stand-in process.
    control = await run_scenario(
        "rpm_control_30"
    )

    reduced = await run_scenario(
        "rpm_halved_15"
    )

    control_result = explain_run(control)
    reduced_result = explain_run(reduced)

    difference = (
        control_result["success_pct"]
        - reduced_result["success_pct"]
    )

    print("\n" + "=" * 55)
    print("COMPARISON")
    print("=" * 55)

    print(
        "Control success:",
        round(control_result["success_pct"], 1),
        "%"
    )

    print(
        "Halved RPM success:",
        round(reduced_result["success_pct"], 1),
        "%"
    )

    print(
        "Success-rate difference:",
        round(difference, 1),
        "percentage points"
    )

    print("\nRaw measurements:", CSV_PATH)

    print(
        "\nInterpretation: This experiment isolates "
        "the simulated RPM setting while holding "
        "other configured workload parameters fixed."
    )


if __name__ == "__main__":
    asyncio.run(main())
