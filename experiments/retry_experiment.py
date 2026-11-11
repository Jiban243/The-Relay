
"""Reproduce harmful retries under simulated RPM exhaustion.

Run:
    python -m experiments.retry_experiment

Uses the stand-in only. No real Groq API calls.
"""

import asyncio
import json

from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from experiments.run_load import (
    FIELDS,
    ROOT,
    append_csv,
    start_standin,
    stop_standin,
)

from src.pipeline import RelayPipeline

from src.retry_policy import (
    RetryOutcome,
    call_with_retry,
    limit_code,
)


async def naive_retries(call):
    """
    Deliberately bad policy:
    ignore Retry-After and retry immediately
    after short fixed delays.
    """

    started = perf_counter()

    attempts = []
    retry_wait_ms = 0.0

    delays = (0.25, 0.50)
    final = None

    for index in range(3):

        attempt_started = perf_counter()

        final = await call()

        attempts.append({
            "attempt": index + 1,
            "status": final.get("status"),
            "http_status": final.get("http_status"),
            "limit_code": limit_code(final),
            "retry_after_s": (
                final.get("rate_headers") or {}
            ).get("retry-after"),
            "elapsed_ms": round(
                (perf_counter() - attempt_started) * 1000,
                2,
            ),
        })

        if final.get("status") == "ok":
            break

        if index < len(delays):

            await asyncio.sleep(delays[index])

            retry_wait_ms += delays[index] * 1000

    stop_reason = (
        "succeeded"
        if final.get("status") == "ok"
        else "naive_attempts_exhausted"
    )

    return RetryOutcome(
        result=final,
        attempts=attempts,
        retries=max(0, len(attempts) - 1),
        retry_wait_ms=round(retry_wait_ms, 2),
        total_ms=round(
            (perf_counter() - started) * 1000,
            2,
        ),
        stop_reason=stop_reason,
    )


def csv_record(run_id, label, started_at, outcome):
    """Convert a retry trial to our existing CSV schema."""

    row = {
        field: ""
        for field in FIELDS
    }

    row.update({
        "request_id": str(uuid4()),
        "scenario": label,
        "source": "stand-in",
        "start_time": started_at,
        "first_token_ms": "",
        "total_ms": outcome.total_ms,
        "status": outcome.result.get(
            "status", "error"
        ),
        "retries": outcome.retries,
        "tokens_in": "",
        "tokens_out": "",
        "retrieval_setting": "not_applicable",
        "answer_grade": "",
        "run_id": run_id,
        "question_id": "rpm_retry_probe",
        "admission_reason": "provider_rpm_rejection",
        "queue_wait_ms": 0,
        "provider_first_content_ms": "",
        "user_visible_first_content_ms": "",
        "model_ms": outcome.total_ms,
        "revisions": 0,
        "peak_queue": 0,
        "peak_in_flight": 1,
        "configured_rpm": 1,
        "configured_rpd": 1000,
        "configured_tpm": 50000,
        "max_in_flight": 2,
        "error_detail": (
            f"{outcome.stop_reason};"
            f"retry_wait_ms={outcome.retry_wait_ms}"
        ),
    })

    return row


async def main():

    run_id = "retry_" + uuid4().hex[:10]

    config = {
        "rpm": 1,
        "rpd": 1000,
        "in_flight": 2,
        "fail_prob": 0.0,
        "timeout_prob": 0.0,
        "tail_prob": 0.0,
    }

    process, log_file, base_url = (
        await start_standin(config, run_id)
    )

    pipeline = None

    try:

        pipeline = RelayPipeline(
            source="stand-in"
        )

        pipeline.url = (
            base_url
            + "/openai/v1/chat/completions"
        )

        messages = [
            {
                "role": "user",
                "content": "Synthetic limit-check probe",
            }
        ]

        async def call():
            return await pipeline._model_call(
                messages,
                max_tokens=256,
            )

        print("\n========== RETRY EXPERIMENT ==========")
        print("Source: stand-in only")
        print("Configured RPM: 1")

        # Consume the single RPM slot.
        warmup = await call()

        if warmup["status"] != "ok":
            raise RuntimeError(
                "Could not prime simulated RPM: "
                f"{warmup['status']} "
                f"{warmup.get('error')}"
            )

        print("\nWarmup request: successful")
        print("Simulated RPM capacity is now exhausted.")

        trials = []

        policies = [
            (
                "retry_no_retry",
                lambda: call_with_retry(
                    call,
                    max_retries=0,
                    deadline_s=2.0,
                ),
            ),
            (
                "retry_naive",
                lambda: naive_retries(call),
            ),
            (
                "retry_deadline_aware",
                lambda: call_with_retry(
                    call,
                    max_retries=2,
                    deadline_s=2.0,
                ),
            ),
        ]

        for label, invoke in policies:

            started_at = datetime.now(
                timezone.utc
            ).isoformat()

            outcome = await invoke()

            trials.append({
                "scenario": label,
                "start_time": started_at,
                "outcome": outcome.to_dict(),
            })

            print(f"\nPolicy: {label}")
            print(
                "Status:",
                outcome.result["status"]
            )
            print(
                "Attempts:",
                len(outcome.attempts)
            )
            print(
                "Retries:",
                outcome.retries
            )
            print(
                "Retry wait:",
                outcome.retry_wait_ms,
                "ms"
            )
            print(
                "Total time:",
                outcome.total_ms,
                "ms"
            )
            print(
                "Stop reason:",
                outcome.stop_reason
            )

        csv_rows = []

        for trial in trials:

            outcome = RetryOutcome(
                **trial["outcome"]
            )

            csv_rows.append(
                csv_record(
                    run_id,
                    trial["scenario"],
                    trial["start_time"],
                    outcome,
                )
            )

        append_csv(csv_rows)

        output = (
            ROOT
            / "results"
            / f"retry_experiment_{run_id}.json"
        )

        output.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "source": "stand-in",
                    "configured_rpm": 1,
                    "warmup_status": warmup["status"],
                    "trials": trials,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        baseline = trials[0]["outcome"]
        naive = trials[1]["outcome"]

        extra_latency = (
            naive["total_ms"]
            - baseline["total_ms"]
        )

        extra_attempts = (
            len(naive["attempts"])
            - len(baseline["attempts"])
        )

        print("\n========== HARMFUL RETRY COMPARISON ==========")

        print(
            "No-retry total:",
            baseline["total_ms"],
            "ms"
        )

        print(
            "Naive-retry total:",
            naive["total_ms"],
            "ms"
        )

        print(
            "Extra latency:",
            round(extra_latency, 2),
            "ms"
        )

        print(
            "Extra provider attempts:",
            extra_attempts
        )

        print("\nDetailed trace:", output)

        print(
            "Request-level records:",
            ROOT / "results" / "results.csv"
        )

    finally:

        if pipeline is not None:
            await pipeline.close()

        stop_standin(
            process,
            log_file,
        )


if __name__ == "__main__":
    asyncio.run(main())
