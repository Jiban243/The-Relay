
import asyncio

from src.retry_policy import call_with_retry


def run(coro):
    return asyncio.run(coro)


def test_success_needs_no_retry():

    calls = 0

    async def fake():
        nonlocal calls
        calls += 1

        return {
            "status": "ok",
            "http_status": 200,
        }

    result = run(call_with_retry(fake))

    assert calls == 1
    assert result.retries == 0
    assert result.stop_reason == "succeeded"


def test_minute_limit_exceeds_deadline():

    calls = 0

    async def fake():
        nonlocal calls
        calls += 1

        return {
            "status": "rate_limited",
            "http_status": 429,
            "rate_headers": {
                "retry-after": "59"
            },
            "error": (
                '{"error":{"code":"rpm"}}'
            ),
        }

    result = run(
        call_with_retry(
            fake,
            deadline_s=1,
        )
    )

    assert calls == 1
    assert result.retries == 0
    assert result.stop_reason == (
        "retry_exceeds_deadline"
    )


def test_daily_limit_never_retries():

    async def fake():
        return {
            "status": "rate_limited",
            "http_status": 429,
            "rate_headers": {
                "retry-after": "1"
            },
            "error": (
                '{"error":{"code":"rpd"}}'
            ),
        }

    result = run(
        call_with_retry(
            fake,
            deadline_s=4,
        )
    )

    assert result.retries == 0
    assert result.stop_reason == (
        "daily_quota_no_retry"
    )


def test_transient_503_retries_once():

    calls = 0

    async def fake():
        nonlocal calls
        calls += 1

        if calls == 1:
            return {
                "status": "error",
                "http_status": 503,
                "text": "",
            }

        return {
            "status": "ok",
            "http_status": 200,
        }

    result = run(
        call_with_retry(
            fake,
            deadline_s=2,
            base_backoff_s=0.01,
        )
    )

    assert calls == 2
    assert result.retries == 1
    assert result.stop_reason == "succeeded"


def test_partial_answer_not_retried():

    calls = 0

    async def fake():
        nonlocal calls
        calls += 1

        return {
            "status": "error",
            "http_status": 503,
            "text": "Partial output",
        }

    result = run(
        call_with_retry(fake)
    )

    assert calls == 1
    assert result.stop_reason == (
        "partial_response_no_retry"
    )
