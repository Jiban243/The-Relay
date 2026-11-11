
"""The Relay: isolated stand-in load experiments.

Commands:
    python -m experiments.run_load --scenario steady
    python -m experiments.run_load --scenario all

Each scenario starts its own stand-in service with fresh
quota counters. This script makes no real Groq API calls.
"""

import argparse
import asyncio
import csv
import json
import os
import socket
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx
import numpy as np

from src.admission import AdmissionController, AdmissionLimits
from src.pipeline import RelayPipeline


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "results.csv"


# Experimental configurations, not measured Groq limits.
SCENARIOS = {
    "steady": {
        "count": 8,
        "interval_s": 2.5,
        "rpm": 30,
        "rpd": 1000,
        "in_flight": 3,
        "queue": 12,
        "wait_s": 2.0,
    },

    "burst": {
        "count": 24,
        "interval_s": 0.01,
        "rpm": 30,
        "rpd": 1000,
        "in_flight": 3,
        "queue": 12,
        "wait_s": 1.5,
    },

    "sustained_overload": {
        "count": 48,
        "interval_s": 0.15,
        "rpm": 30,
        "rpd": 1000,
        "in_flight": 2,
        "queue": 8,
        "wait_s": 1.0,
        "fail_prob": 0.06,
        "timeout_prob": 0.03,
        "tail_prob": 0.12,
    },

    "rpm_halved": {
        "count": 24,
        "interval_s": 0.06,
        "rpm": 15,
        "rpd": 1000,
        "in_flight": 3,
        "queue": 12,
        "wait_s": 1.5,
    },

    "daily_exhausted": {
        "count": 12,
        "interval_s": 0.09,
        "rpm": 30,
        "rpd": 3,
        "in_flight": 3,
        "queue": 12,
        "wait_s": 1.5,
    },
}


QUESTIONS = [
    (
        "attention",
        "According to Attention Is All You Need, "
        "how is scaled dot-product attention computed?",
    ),
    (
        "moshi",
        "According to Moshi, how does full-duplex "
        "spoken dialogue work?",
    ),
    (
        "hnet",
        "According to H-Net, how does dynamic "
        "chunking operate?",
    ),
]


# Mandatory results.csv columns appear first.
# Additional columns preserve experimental evidence.
FIELDS = [
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
    "run_id",
    "question_id",
    "admission_reason",
    "queue_wait_ms",
    "provider_first_content_ms",
    "user_visible_first_content_ms",
    "retrieval_ms",
    "model_ms",
    "revisions",
    "peak_queue",
    "peak_in_flight",
    "duration_s",
    "offered_rate_per_min",
    "configured_rpm",
    "configured_rpd",
    "configured_tpm",
    "max_in_flight",
    "error_detail",
]


def free_port():
    """Find an available local TCP port."""

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def start_standin(config, run_id):
    """Start a fresh stand-in service for one experiment."""

    port = free_port()

    env = os.environ.copy()
    env.pop("GROQ_API_KEY", None)

    env.update({
        "SIM_RPM": str(config["rpm"]),
        "SIM_RPD": str(config["rpd"]),

        # Raised for these laboratory scenarios
        # to isolate RPM and concurrency effects.
        # This is NOT Groq's measured TPM.
        "SIM_TPM": "50000",
        "SIM_TPD": "500000",

        "SIM_MAX_IN_FLIGHT": str(
            config["in_flight"]
        ),

        "SIM_FAILURE_PROBABILITY": str(
            config.get("fail_prob", 0.0)
        ),

        "SIM_TIMEOUT_PROBABILITY": str(
            config.get("timeout_prob", 0.0)
        ),

        "SIM_TAIL_PROBABILITY": str(
            config.get("tail_prob", 0.0)
        ),

        "SIM_LOAD_SLOWDOWN": "0.20",
        "SIM_SEED": "42",
    })

    log_path = (
        ROOT / "results" / f"standin_{run_id}.log"
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = log_path.open(
        "w", encoding="utf-8"
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.standin_service:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    base_url = f"http://127.0.0.1:{port}"

    try:
        async with httpx.AsyncClient(
            timeout=1
        ) as client:

            for _ in range(70):

                if process.poll() is not None:
                    raise RuntimeError(
                        f"Stand-in exited. See {log_path}"
                    )

                try:
                    response = await client.get(
                        base_url + "/health"
                    )

                    if response.status_code == 200:
                        return (
                            process,
                            log_file,
                            base_url,
                        )

                except httpx.HTTPError:
                    pass

                await asyncio.sleep(0.1)

        raise RuntimeError(
            f"Stand-in failed to start. See {log_path}"
        )

    except BaseException:
        process.terminate()
        process.wait(timeout=5)
        log_file.close()
        raise


def stop_standin(process, log_file):
    """Shut down the isolated simulation process."""

    if process.poll() is None:

        process.terminate()

        try:
            process.wait(timeout=5)

        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    log_file.close()


def percentile(rows, key, p):
    """Calculate a percentile from recorded measurements."""

    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None
    ]

    if not values:
        return None

    return round(
        float(np.percentile(values, p)),
        2,
    )


def append_csv(rows):
    """Append raw request results without deleting older runs."""

    CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    needs_header = (
        not CSV_PATH.exists()
        or CSV_PATH.stat().st_size == 0
    )

    if not needs_header:

        with CSV_PATH.open(
            newline="", encoding="utf-8"
        ) as file:

            existing_header = next(
                csv.reader(file),
                [],
            )

        if existing_header != FIELDS:
            raise RuntimeError(
                "Existing results.csv has a different header. "
                "Back it up before changing the schema."
            )

    with CSV_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )

        if needs_header:
            writer.writeheader()

        writer.writerows(rows)


async def run_scenario(name):

    config = SCENARIOS[name]

    run_id = (
        f"{name}_{uuid4().hex[:10]}"
    )

    process, log_file, base_url = (
        await start_standin(config, run_id)
    )

    controller = AdmissionController(
        AdmissionLimits(
            rpm=config["rpm"],
            rpd=config["rpd"],
            tpm=50000,
            tpd=500000,
            max_in_flight=config["in_flight"],
            max_queue=config["queue"],
        )
    )

    pipeline = None
    arrival_offsets = []
    run_started = perf_counter()

    try:

        pipeline = RelayPipeline(
            source="stand-in"
        )

        # Point only this pipeline to its isolated server.
        pipeline.url = (
            base_url
            + "/openai/v1/chat/completions"
        )

        async def one_request(index):

            await asyncio.sleep(
                index * config["interval_s"]
            )

            arrived = perf_counter()

            arrival_offsets.append(
                arrived - run_started
            )

            question_id, question = QUESTIONS[
                index % len(QUESTIONS)
            ]

            row = {
                "request_id": str(uuid4()),
                "scenario": name,
                "source": "stand-in",
                "start_time": datetime.now(
                    timezone.utc
                ).isoformat(),
                "first_token_ms": None,
                "total_ms": None,
                "status": "error",
                "retries": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "retrieval_setting": "small",

                # Synthetic responses cannot receive
                # factual answer-quality grades.
                "answer_grade": "",

                "run_id": run_id,
                "question_id": question_id,
                "admission_reason": "",
                "queue_wait_ms": 0.0,
                "provider_first_content_ms": None,
                "user_visible_first_content_ms": None,
                "retrieval_ms": None,
                "model_ms": None,
                "revisions": 0,
                "configured_rpm": config["rpm"],
                "configured_rpd": config["rpd"],
                "configured_tpm": 50000,
                "max_in_flight": config["in_flight"],
                "error_detail": "",
            }

            admitted = False

            try:

                # Reserve enough capacity for one
                # estimated model call.
                lease = await controller.acquire(
                    estimated_tokens=1700,
                    max_wait_s=config["wait_s"],
                )

                row["queue_wait_ms"] = (
                    lease.queue_wait_ms
                )

                row["admission_reason"] = (
                    lease.reason
                )

                if not lease.accepted:

                    if lease.reason == "daily_exhausted":
                        row["status"] = "rate_limited"

                    elif lease.reason == "queue_timeout":
                        row["status"] = "timeout"

                    else:
                        row["status"] = "error"

                    row["error_detail"] = lease.reason

                    return row

                admitted = True

                result = await pipeline.answer(
                    question=question,
                    setting="small",
                    scenario=name,
                    allow_revision=False,
                )

                row["status"] = result["status"]
                row["retries"] = result["retries"]

                row["revisions"] = result.get(
                    "revisions", 0
                )

                row["tokens_in"] = result["tokens_in"]
                row["tokens_out"] = result["tokens_out"]

                row["retrieval_ms"] = (
                    result["stages_ms"].get("retrieval")
                )

                row["model_ms"] = (
                    result["stages_ms"].get("model_calls")
                )

                provider_first = result.get(
                    "provider_first_content_ms"
                )

                if provider_first is not None:

                    # Model first-content latency
                    # including admission waiting.
                    row["first_token_ms"] = round(
                        lease.queue_wait_ms
                        + provider_first,
                        2,
                    )

                    row["provider_first_content_ms"] = (
                        row["first_token_ms"]
                    )

                if result["status"] == "ok":

                    # The pipeline buffers the answer.
                    # User-visible content becomes ready
                    # only after the pipeline finishes.
                    row["user_visible_first_content_ms"] = (
                        round(
                            (perf_counter() - arrived)
                            * 1000,
                            2,
                        )
                    )

                elif result.get("model_calls"):

                    row["error_detail"] = str(
                        result["model_calls"][-1].get(
                            "error"
                        ) or "model_failure"
                    )[:200]

            except Exception as exc:

                row["status"] = "error"

                row["error_detail"] = (
                    f"{type(exc).__name__}: {exc}"
                )[:200]

            finally:

                if admitted:
                    await controller.release()

                row["total_ms"] = round(
                    (perf_counter() - arrived) * 1000,
                    2,
                )

            return row

        # Submit all scheduled requests concurrently.
        rows = await asyncio.gather(
            *[
                one_request(i)
                for i in range(config["count"])
            ]
        )

        duration_s = round(
            perf_counter() - run_started,
            3,
        )

        arrival_span = (
            max(arrival_offsets)
            - min(arrival_offsets)
        )

        # Use a minimum one-second arrival window
        # for interpreting very short bursts.
        offered_rate = round(
            60 * config["count"]
            / max(1.0, arrival_span),
            2,
        )

        metrics = await controller.snapshot()

        async with httpx.AsyncClient(
            timeout=3
        ) as client:

            response = await client.get(
                base_url + "/metrics"
            )

            sim_metrics = response.json()

        # Add scenario-wide measurements to each row,
        # allowing a reviewer to reconstruct summaries.
        for row in rows:

            row["peak_queue"] = (
                metrics["peak_queue"]
            )

            row["peak_in_flight"] = (
                metrics["peak_in_flight"]
            )

            row["duration_s"] = duration_s

            row["offered_rate_per_min"] = (
                offered_rate
            )

        append_csv(rows)

        successful = [
            row for row in rows
            if row["status"] == "ok"
        ]

        summary = {
            "run_id": run_id,
            "scenario": name,
            "source": "stand-in",
            "requests": len(rows),
            "duration_s": duration_s,
            "offered_rate_per_min": offered_rate,
            "config": config,
            "app_peak_queue": metrics["peak_queue"],
            "app_peak_in_flight": (
                metrics["peak_in_flight"]
            ),
            "standin_metrics": sim_metrics,

            "status_counts": {
                status: sum(
                    row["status"] == status
                    for row in rows
                )
                for status in (
                    "ok",
                    "rate_limited",
                    "timeout",
                    "error",
                )
            },

            "model_ttft_ms": {
                f"p{p}": percentile(
                    successful,
                    "first_token_ms",
                    p,
                )
                for p in (50, 95, 99)
            },

            "successful_total_ms": {
                f"p{p}": percentile(
                    successful,
                    "total_ms",
                    p,
                )
                for p in (50, 95, 99)
            },

            "warning": (
                "Stand-in timings and token counts "
                "are simulated. TTFT measures first "
                "model content including queue wait. "
                "Buffered user-visible first content "
                "is recorded separately. Small-sample "
                "p95/p99 are descriptive only. "
                "Laboratory TPM=50000 intentionally "
                "differs from real observed TPM=8000."
            ),
        }

        summary_path = (
            ROOT
            / "results"
            / f"load_summary_{run_id}.json"
        )

        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print(f"\n========== {name.upper()} ==========")

        print("Run ID:", run_id)

        print("Requests:", len(rows))

        print("Duration:", duration_s, "seconds")

        print(
            "Offered rate:",
            offered_rate,
            "requests/minute",
        )

        print(
            "Status counts:",
            summary["status_counts"],
        )

        print(
            "Peak queue:",
            summary["app_peak_queue"],
        )

        print(
            "Peak in-flight:",
            summary["app_peak_in_flight"],
        )

        print(
            "Model first-token p50/p95/p99:",
            summary["model_ttft_ms"],
        )

        print(
            "Total latency p50/p95/p99:",
            summary["successful_total_ms"],
        )

        print("Summary:", summary_path)

        return summary

    finally:

        if pipeline is not None:
            await pipeline.close()

        stop_standin(process, log_file)


async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        choices=["all", *SCENARIOS],
        default="steady",
    )

    args = parser.parse_args()

    selected = (
        list(SCENARIOS)
        if args.scenario == "all"
        else [args.scenario]
    )

    print(
        "Running stand-in experiments only. "
        "No real Groq requests will be made."
    )

    for name in selected:
        await run_scenario(name)

    print("\nRaw results saved to:", CSV_PATH)


if __name__ == "__main__":
    asyncio.run(main())
