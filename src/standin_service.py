
import asyncio
import json
import math
import os
import random
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = (
    ROOT / "results" / "real_service_calibration.jsonl"
)

app = FastAPI(title="The Relay Stand-in Model")


def env_int(name, default):
    return int(os.getenv(name, default))


def env_float(name, default):
    return float(os.getenv(name, default))


# Defaults:
# RPM/TPD are documented free-tier reference limits.
# RPD/TPM also appeared in our observed real headers.
# Concurrency and failure behavior are simulation choices.
CONFIG = {
    "rpm": env_int("SIM_RPM", 30),
    "tpm": env_int("SIM_TPM", 8000),
    "rpd": env_int("SIM_RPD", 1000),
    "tpd": env_int("SIM_TPD", 200000),
    "max_in_flight": env_int("SIM_MAX_IN_FLIGHT", 8),
    "load_slowdown": env_float("SIM_LOAD_SLOWDOWN", 0.20),
    "failure_probability": env_float(
        "SIM_FAILURE_PROBABILITY", 0.0
    ),
    "timeout_probability": env_float(
        "SIM_TIMEOUT_PROBABILITY", 0.0
    ),
    "tail_probability": env_float(
        "SIM_TAIL_PROBABILITY", 0.0
    ),
    "tail_multiplier": env_float(
        "SIM_TAIL_MULTIPLIER", 3.0
    ),
    "timeout_ms": env_int("SIM_TIMEOUT_MS", 3000),
    "seed": env_int("SIM_SEED", 42),
}

if CONFIG["max_in_flight"] < 1:
    raise ValueError("SIM_MAX_IN_FLIGHT must be positive")

for name in (
    "failure_probability",
    "timeout_probability",
    "tail_probability",
):
    if not 0 <= CONFIG[name] <= 1:
        raise ValueError(f"{name} must be between 0 and 1")

if (
    CONFIG["failure_probability"]
    + CONFIG["timeout_probability"]
    > 1
):
    raise ValueError("Failure and timeout probabilities exceed 1")


def load_calibration():
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Missing calibration data: {CALIBRATION_FILE}"
        )

    samples = []

    with CALIBRATION_FILE.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if (
                record.get("status") == "ok"
                and record.get("source") == "real"
                and record.get("first_token_ms") is not None
                and record.get("total_ms") is not None
            ):
                samples.append({
                    "first_token_ms": float(
                        record["first_token_ms"]
                    ),
                    "total_ms": float(record["total_ms"]),
                    "tokens_in": record.get("tokens_in"),
                    "tokens_out": record.get("tokens_out"),
                })

    if not samples:
        raise ValueError(
            "No successful real calibration samples found"
        )

    return samples


SAMPLES = load_calibration()
RNG = random.Random(CONFIG["seed"])


class ChatRequest(BaseModel):
    model: str = "stand-in"
    messages: list[dict]
    stream: bool = True
    max_completion_tokens: int = Field(default=256, ge=1)


class LimitState:
    def __init__(self):
        self.lock = asyncio.Lock()

        now = time.monotonic()

        self.minute_start = now
        self.day_start = now

        self.minute_requests = 0
        self.day_requests = 0

        self.minute_tokens = 0
        self.day_tokens = 0

        self.in_flight = 0
        self.peak_in_flight = 0

    def refresh(self):
        now = time.monotonic()

        if now - self.minute_start >= 60:
            self.minute_start = now
            self.minute_requests = 0
            self.minute_tokens = 0

        if now - self.day_start >= 86400:
            self.day_start = now
            self.day_requests = 0
            self.day_tokens = 0

    def headers(self):
        return {
            "x-ratelimit-limit-requests": str(CONFIG["rpd"]),
            "x-ratelimit-limit-tokens": str(CONFIG["tpm"]),
            "x-ratelimit-remaining-requests": str(
                max(0, CONFIG["rpd"] - self.day_requests)
            ),
            "x-ratelimit-remaining-tokens": str(
                max(0, CONFIG["tpm"] - self.minute_tokens)
            ),
        }

    async def admit(self, reserved_tokens):
        async with self.lock:
            self.refresh()

            now = time.monotonic()

            if self.day_requests >= CONFIG["rpd"]:
                return False, "rpd", max(
                    1, math.ceil(
                        86400 - (now - self.day_start)
                    )
                ), 0, self.headers()

            if (
                self.day_tokens + reserved_tokens
                > CONFIG["tpd"]
            ):
                return False, "tpd", max(
                    1, math.ceil(
                        86400 - (now - self.day_start)
                    )
                ), 0, self.headers()

            if self.minute_requests >= CONFIG["rpm"]:
                return False, "rpm", max(
                    1, math.ceil(
                        60 - (now - self.minute_start)
                    )
                ), 0, self.headers()

            if (
                self.minute_tokens + reserved_tokens
                > CONFIG["tpm"]
            ):
                return False, "tpm", max(
                    1, math.ceil(
                        60 - (now - self.minute_start)
                    )
                ), 0, self.headers()

            if self.in_flight >= CONFIG["max_in_flight"]:
                return False, "concurrency", 1, 0, self.headers()

            self.minute_requests += 1
            self.day_requests += 1

            self.minute_tokens += reserved_tokens
            self.day_tokens += reserved_tokens

            self.in_flight += 1

            self.peak_in_flight = max(
                self.peak_in_flight,
                self.in_flight,
            )

            return (
                True,
                None,
                0,
                self.in_flight,
                self.headers(),
            )

    async def release(self):
        async with self.lock:
            self.in_flight = max(0, self.in_flight - 1)


STATE = LimitState()


def estimate_input_tokens(messages):
    # Approximation for admission control, not a
    # claim about Groq's actual tokenizer.
    characters = sum(
        len(str(message.get("content") or ""))
        for message in messages
    )

    return max(1, math.ceil(characters / 4))


def choose_latency(in_flight):
    # Bootstrap an observed first-token / total-time pair.
    # Only successful real measurements are sampled.
    sample = RNG.choice(SAMPLES)

    scale = 1.0 + (
        CONFIG["load_slowdown"] * (in_flight - 1)
    )

    # Mild synthetic variation around sampled observations.
    scale *= RNG.uniform(0.90, 1.10)

    if RNG.random() < CONFIG["tail_probability"]:
        scale *= CONFIG["tail_multiplier"]

    first_ms = max(
        1.0,
        sample["first_token_ms"] * scale,
    )

    total_ms = max(
        first_ms + 1.0,
        sample["total_ms"] * scale,
    )

    return first_ms, total_ms, sample


def sse(data):
    return f"data: {json.dumps(data)}\n\n"


def response_text():
    return (
        "SIMULATED MODEL RESPONSE. "
        "This is synthetic content for latency, "
        "streaming, quota and failure testing only. "
        "It must not be graded as a factual answer "
        "to a research-paper question."
    )


def usage_object(input_tokens, output_tokens):
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "relay-stand-in",
        "calibration_samples": len(SAMPLES),
        "config": CONFIG,
    }


@app.get("/metrics")
async def metrics():
    async with STATE.lock:
        STATE.refresh()

        return {
            "minute_requests": STATE.minute_requests,
            "day_requests": STATE.day_requests,
            "minute_tokens_reserved": STATE.minute_tokens,
            "day_tokens_reserved": STATE.day_tokens,
            "in_flight": STATE.in_flight,
            "peak_in_flight": STATE.peak_in_flight,
        }


@app.post("/openai/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    input_tokens = estimate_input_tokens(
        request.messages
    )

    # Deliberately conservative token reservation.
    reserved_tokens = (
        input_tokens + request.max_completion_tokens
    )

    admitted, reason, retry_after, in_flight, headers = (
        await STATE.admit(reserved_tokens)
    )

    if not admitted:
        headers = {
            **headers,
            "retry-after": str(retry_after),
        }

        return JSONResponse(
            status_code=429,
            headers=headers,
            content={
                "error": {
                    "type": "rate_limit_error",
                    "code": reason,
                    "message": (
                        f"Simulated {reason} limit reached"
                    ),
                }
            },
        )

    first_ms, total_ms, sample = choose_latency(
        in_flight
    )

    # These probabilities are deliberately injected,
    # not inferred from the 3 successful real samples.
    fate = RNG.random()

    if fate < CONFIG["timeout_probability"]:
        try:
            await asyncio.sleep(
                CONFIG["timeout_ms"] / 1000
            )
        finally:
            await STATE.release()

        return JSONResponse(
            status_code=504,
            headers=headers,
            content={
                "error": {
                    "type": "timeout",
                    "message": "Simulated service timeout",
                }
            },
        )

    if fate < (
        CONFIG["timeout_probability"]
        + CONFIG["failure_probability"]
    ):
        try:
            await asyncio.sleep(first_ms / 1000)
        finally:
            await STATE.release()

        return JSONResponse(
            status_code=503,
            headers=headers,
            content={
                "error": {
                    "type": "server_error",
                    "message": "Simulated model failure",
                }
            },
        )

    request_id = f"sim-{uuid4()}"

    output_tokens = min(
        request.max_completion_tokens,
        int(sample["tokens_out"] or 100),
    )

    answer = response_text()

    if not request.stream:
        try:
            await asyncio.sleep(total_ms / 1000)

            return JSONResponse(
                status_code=200,
                headers=headers,
                content={
                    "id": request_id,
                    "object": "chat.completion",
                    "model": "relay-stand-in",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": answer,
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": usage_object(
                        input_tokens,
                        output_tokens,
                    ),
                },
            )
        finally:
            await STATE.release()

    async def generate():
        try:
            # Initial role event, without content.
            yield sse({
                "id": request_id,
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }],
            })

            await asyncio.sleep(first_ms / 1000)

            words = answer.split()
            group_size = max(1, math.ceil(len(words) / 6))

            chunks = [
                " ".join(words[i:i + group_size]) + " "
                for i in range(0, len(words), group_size)
            ]

            remaining_seconds = max(
                0.0,
                (total_ms - first_ms) / 1000,
            )

            for index, piece in enumerate(chunks):
                yield sse({
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": None,
                    }],
                })

                if index < len(chunks) - 1:
                    await asyncio.sleep(
                        remaining_seconds / max(
                            1, len(chunks) - 1
                        )
                    )

            yield sse({
                "id": request_id,
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            })

            yield sse({
                "id": request_id,
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": usage_object(
                    input_tokens,
                    output_tokens,
                ),
            })

            yield "data: [DONE]\n\n"

        finally:
            await STATE.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            **headers,
            "Cache-Control": "no-cache",
        },
    )
