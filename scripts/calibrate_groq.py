
import asyncio
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx
import numpy as np
from dotenv import load_dotenv

from src.retriever import BM25Retriever
from src.evidence import make_grounded_messages


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "real_service_calibration.jsonl"

URL = "https://api.groq.com/openai/v1/chat/completions"


# Three different workloads.
# Increased output budgets to avoid truncated completions.
TASKS = [
    {
        "id": "attention",
        "question": (
            "According to Attention Is All You Need, "
            "how is scaled dot-product attention computed? "
            "Answer in one sentence."
        ),
        "max_tokens": 768,
    },
    {
        "id": "moshi",
        "question": (
            "According to Moshi, how does the model "
            "support full-duplex spoken dialogue? "
            "Answer in two or three sentences."
        ),
        "max_tokens": 768,
    },
    {
        "id": "hnet",
        "question": (
            "According to H-Net, how does dynamic "
            "chunking operate? Explain in four concise bullets."
        ),
        "max_tokens": 1024,
    },
]


async def measure(client, api_key, model, task, messages, run_id):

    record = {
        "request_id": str(uuid4()),
        "run_id": run_id,
        "scenario": "real_calibration",
        "source": "real",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "question_id": task["id"],
        "retrieval_setting": "small",
        "max_completion_tokens": task["max_tokens"],
        "reasoning_effort": "low",
        "status": "error",
        "http_status": None,
        "first_event_ms": None,
        "first_token_ms": None,
        "last_content_ms": None,
        "total_ms": None,
        "tokens_in": None,
        "tokens_out": None,
        "reasoning_tokens": None,
        "finish_reason": None,
        "piece_count": 0,
        "retries": 0,
        "headers": {},
        "answer": "",
        "error": None,
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_completion_tokens": task["max_tokens"],

        # Reduce reasoning-token consumption.
        "reasoning_effort": "low",

        # Hide reasoning content from the response.
        # Reasoning tokens may still be consumed.
        "include_reasoning": False,

        # Stream output and request token-usage reporting.
        "stream": True,
        "stream_options": {
            "include_usage": True
        },
    }

    start = perf_counter()
    pieces = []

    try:
        async with client.stream(
            "POST",
            URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:

            record["http_status"] = response.status_code

            record["headers"] = {
                k.lower(): v
                for k, v in response.headers.items()
                if k.lower().startswith("x-ratelimit")
                or k.lower() == "retry-after"
            }

            if response.status_code >= 400:
                body = await response.aread()

                record["status"] = (
                    "rate_limited"
                    if response.status_code == 429
                    else "error"
                )

                record["error"] = body.decode(
                    errors="replace"
                )[:500]

            else:
                async for line in response.aiter_lines():

                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()

                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    elapsed = (perf_counter() - start) * 1000

                    if record["first_event_ms"] is None:
                        record["first_event_ms"] = round(
                            elapsed, 2
                        )

                    if "error" in event:
                        record["error"] = str(
                            event["error"]
                        )[:500]
                        continue

                    choices = event.get("choices") or []

                    if choices:
                        choice = choices[0]

                        delta = choice.get("delta") or {}

                        piece = delta.get("content") or ""

                        if piece:
                            if record["first_token_ms"] is None:
                                record["first_token_ms"] = round(
                                    elapsed, 2
                                )

                            record["last_content_ms"] = round(
                                elapsed, 2
                            )

                            record["piece_count"] += 1
                            pieces.append(piece)

                        if choice.get("finish_reason"):
                            record["finish_reason"] = (
                                choice["finish_reason"]
                            )

                    usage = event.get("usage")

                    if not usage:
                        usage = (
                            event.get("x_groq") or {}
                        ).get("usage")

                    if usage:
                        record["tokens_in"] = usage.get(
                            "prompt_tokens"
                        )

                        record["tokens_out"] = usage.get(
                            "completion_tokens"
                        )

                        details = usage.get(
                            "completion_tokens_details"
                        ) or {}

                        record["reasoning_tokens"] = details.get(
                            "reasoning_tokens"
                        )

                record["answer"] = "".join(pieces)

                if record["error"]:
                    record["status"] = "error"

                elif (
                    record["answer"].strip()
                    and record["finish_reason"] != "length"
                ):
                    record["status"] = "ok"

                else:
                    record["status"] = "error"
                    record["error"] = (
                        "Empty or truncated completion"
                    )

    except httpx.TimeoutException as exc:
        record["status"] = "timeout"
        record["error"] = type(exc).__name__

    except httpx.HTTPError as exc:
        record["status"] = "error"
        record["error"] = type(exc).__name__

    finally:
        record["total_ms"] = round(
            (perf_counter() - start) * 1000,
            2
        )

        OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with OUTPUT.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    return record


def summarize(records):

    print("\n========== CALIBRATION SUMMARY ==========")

    print("Total requests:", len(records))

    print(
        "Outcomes:",
        dict(Counter(
            r["status"] for r in records
        ))
    )

    for field in ("first_token_ms", "total_ms"):

        values = [
            r[field]
            for r in records
            if r["status"] == "ok"
            and r[field] is not None
        ]

        if not values:
            print(f"{field}: not measured")
            continue

        print(f"\n{field} (n={len(values)})")

        print(
            f"p50: {np.percentile(values, 50):.2f} ms"
        )

        print(
            f"p95: {np.percentile(values, 95):.2f} ms"
        )

        print(
            f"p99: {np.percentile(values, 99):.2f} ms"
        )

    total_in = sum(
        r["tokens_in"] or 0
        for r in records
    )

    total_out = sum(
        r["tokens_out"] or 0
        for r in records
    )

    total_reasoning = sum(
        r["reasoning_tokens"] or 0
        for r in records
    )

    print("\nReported input tokens:", total_in)

    print("Reported output tokens:", total_out)

    print("Reported reasoning tokens:", total_reasoning)

    print("Raw data:", OUTPUT)

    print(
        "\nNote: Tail percentiles from this small "
        "sample are preliminary."
    )


async def main():

    load_dotenv(ROOT / ".env")

    api_key = os.getenv("GROQ_API_KEY")

    model = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b"
    )

    if not api_key:
        raise SystemExit("GROQ_API_KEY missing")

    retriever = BM25Retriever(
        ROOT / "data" / "processed" / "pages.jsonl"
    )

    prepared_tasks = []

    for task in TASKS:

        passages = retriever.search(
            task["question"],
            setting="small",
        )

        if not passages:
            raise RuntimeError(
                f"No passage found for {task['id']}"
            )

        messages = make_grounded_messages(
            task["question"],
            passages[:1],
        )

        prepared_tasks.append(
            (task, messages)
        )

    run_id = str(uuid4())

    records = []

    total_requests = len(prepared_tasks)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(
            45.0,
            connect=10.0
        )
    ) as client:

        # Only one request per question.
        # Three requests in total.
        for task, messages in prepared_tasks:

            print(
                f"\nRequest {len(records) + 1}/{total_requests}"
                f" | {task['id']}"
            )

            result = await measure(
                client,
                api_key,
                model,
                task,
                messages,
                run_id,
            )

            records.append(result)

            print(
                "Status:",
                result["status"]
            )

            print(
                "HTTP status:",
                result["http_status"]
            )

            print(
                "First token:",
                result["first_token_ms"],
                "ms"
            )

            print(
                "Total time:",
                result["total_ms"],
                "ms"
            )

            print(
                "Tokens:",
                result["tokens_in"],
                "+",
                result["tokens_out"]
            )

            print(
                "Reasoning tokens:",
                result["reasoning_tokens"]
            )

            print(
                "Finish reason:",
                result["finish_reason"]
            )

            print(
                "Answer preview:",
                result["answer"][:250]
            )

            print(
                "Remaining TPM:",
                result["headers"].get(
                    "x-ratelimit-remaining-tokens",
                    "unknown",
                )
            )

            if result["error"]:
                print(
                    "Error:",
                    result["error"]
                )

            # Do not retry rate-limited requests.
            if result["status"] == "rate_limited":
                print(
                    "Rate limit encountered. "
                    "Stopping calibration."
                )

                summarize(records)
                return

            remaining = result["headers"].get(
                "x-ratelimit-remaining-tokens"
            )

            if remaining is not None:
                try:
                    if int(remaining) < 700:
                        print(
                            "Low TPM remaining. "
                            "Stopping to protect quota."
                        )

                        summarize(records)
                        return

                except ValueError:
                    pass

            # Spread out the real requests.
            if len(records) < total_requests:
                await asyncio.sleep(12)

    summarize(records)


if __name__ == "__main__":
    asyncio.run(main())
