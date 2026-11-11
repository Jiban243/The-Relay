
"""Deadline-aware model-call retry policy."""

import asyncio
import json

from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Awaitable, Callable


@dataclass
class RetryOutcome:
    result: dict
    attempts: list[dict]
    retries: int
    retry_wait_ms: float
    total_ms: float
    stop_reason: str

    def to_dict(self):
        return asdict(self)


def limit_code(result):
    """Extract the limit type from an API error."""

    raw = result.get("error")

    if not isinstance(raw, str):
        return None

    try:
        parsed = json.loads(raw)
        detail = parsed.get("error") or {}

        if isinstance(detail, dict):
            return detail.get("code")

    except (ValueError, TypeError):
        pass

    return None


def retry_after_seconds(result):
    """Read a numerical Retry-After header."""

    headers = result.get("rate_headers") or {}
    raw = headers.get("retry-after")

    if raw is None:
        return None

    try:
        return max(0.0, float(raw))

    except (TypeError, ValueError):
        return None


async def call_with_retry(
    call: Callable[[], Awaitable[dict]],
    *,
    deadline_s=2.0,
    max_retries=1,
    base_backoff_s=0.2,
    minimum_call_budget_s=0.1,
):
    """
    Retry only when useful and safe.

    A retry must fit inside the remaining deadline.

    Daily quota exhaustion is never retried.

    Responses with partial content are not
    automatically replayed.
    """

    if (
        deadline_s <= 0
        or max_retries < 0
        or base_backoff_s < 0
    ):
        raise ValueError(
            "Invalid retry policy configuration"
        )

    started = perf_counter()

    attempts = []
    wait_ms = 0.0

    last = {
        "status": "error",
        "error": "not_attempted",
    }

    stop_reason = "unknown"

    for attempt_number in range(max_retries + 1):

        remaining = deadline_s - (
            perf_counter() - started
        )

        if remaining <= 0:
            stop_reason = "deadline_exhausted"
            break

        attempt_started = perf_counter()

        try:
            last = await asyncio.wait_for(
                call(),
                timeout=remaining,
            )

        except asyncio.TimeoutError:
            last = {
                "status": "timeout",
                "http_status": None,
                "text": "",
                "error": "client_deadline_exceeded",
                "rate_headers": {},
            }

        attempts.append({
            "attempt": attempt_number + 1,
            "status": last.get("status"),
            "http_status": last.get("http_status"),
            "elapsed_ms": round(
                (perf_counter() - attempt_started) * 1000,
                2,
            ),
            "limit_code": limit_code(last),
            "retry_after_s": retry_after_seconds(last),
        })

        if last.get("status") == "ok":
            stop_reason = "succeeded"
            break

        if attempt_number >= max_retries:
            stop_reason = "retry_budget_exhausted"
            break

        status = last.get("status")
        http_status = last.get("http_status")

        # Never automatically replay partial responses.
        if (
            last.get("text")
            or last.get("first_content_ms") is not None
        ):
            stop_reason = "partial_response_no_retry"
            break

        if (
            status == "rate_limited"
            and http_status == 429
        ):

            code = limit_code(last)

            if code in ("rpd", "tpd"):
                stop_reason = "daily_quota_no_retry"
                break

            retry_after = retry_after_seconds(last)

            if retry_after is None:
                stop_reason = "unknown_reset_no_retry"
                break

            wait_s = max(
                retry_after,
                base_backoff_s * (2 ** attempt_number),
            )

        elif (
            status == "error"
            and http_status in (502, 503)
        ):

            wait_s = (
                base_backoff_s * (2 ** attempt_number)
            )

        else:
            stop_reason = "non_retryable_error"
            break

        remaining = deadline_s - (
            perf_counter() - started
        )

        if (
            wait_s + minimum_call_budget_s
            > remaining
        ):
            stop_reason = "retry_exceeds_deadline"
            break

        await asyncio.sleep(wait_s)

        wait_ms += wait_s * 1000

    return RetryOutcome(
        result=last,
        attempts=attempts,
        retries=max(0, len(attempts) - 1),
        retry_wait_ms=round(wait_ms, 2),
        total_ms=round(
            (perf_counter() - started) * 1000,
            2,
        ),
        stop_reason=stop_reason,
    )
