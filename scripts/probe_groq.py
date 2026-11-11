
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "real_service_probe.jsonl"

URL = "https://api.groq.com/openai/v1/chat/completions"


async def main():
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    if not api_key:
        raise SystemExit("GROQ_API_KEY is missing in .env")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: measurement ok",
            }
        ],
        "max_completion_tokens": 256,
        "stream": True,
        "stream_options": {
            "include_usage": True
        },
    }

    result = {
        "probe_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "source": "real",
        "status": "error",
        "http_status": None,
        "first_token_ms": None,
        "total_ms": None,
        "tokens_in": None,
        "tokens_out": None,
        "headers": {},
        "answer": "",
        "error": None,
    }

    start = perf_counter()
    pieces = []

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(45.0, connect=10.0)
        ) as client:

            async with client.stream(
                "POST",
                URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:

                result["http_status"] = response.status_code

                # Record provider rate-limit metadata.
                result["headers"] = {
                    key.lower(): value
                    for key, value in response.headers.items()
                    if key.lower().startswith("x-ratelimit")
                    or key.lower() == "retry-after"
                }

                if response.status_code >= 400:

                    raw = await response.aread()

                    result["status"] = (
                        "rate_limited"
                        if response.status_code == 429
                        else "error"
                    )

                    result["error"] = raw.decode(
                        errors="replace"
                    )[:600]

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

                        choices = event.get("choices") or []

                        if choices:
                            delta = choices[0].get("delta") or {}
                            piece = delta.get("content") or ""

                            if piece:
                                if result["first_token_ms"] is None:
                                    result["first_token_ms"] = round(
                                        (perf_counter() - start) * 1000,
                                        2,
                                    )

                                pieces.append(piece)

                        usage = event.get("usage")

                        if not usage:
                            usage = (
                                event.get("x_groq") or {}
                            ).get("usage")

                        if usage:
                            result["tokens_in"] = usage.get(
                                "prompt_tokens"
                            )
                            result["tokens_out"] = usage.get(
                                "completion_tokens"
                            )

                    result["answer"] = "".join(pieces)
                    result["status"] = (
                        "ok" if result["answer"] else "error"
                    )

    except httpx.TimeoutException as error:
        result["status"] = "timeout"
        result["error"] = type(error).__name__

    except httpx.HTTPError as error:
        result["status"] = "error"
        result["error"] = type(error).__name__

    finally:
        result["total_ms"] = round(
            (perf_counter() - start) * 1000,
            2,
        )

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)

        with OUTPUT.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result) + "\n")

        print("\n======= GROQ REAL SERVICE PROBE =======")
        print(json.dumps(result, indent=2))
        print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
