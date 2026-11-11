
"""FIFO admission control for The Relay.

A lease reserves capacity for one planned model call.
The controller uses conservative token estimates.

This is our application policy, not an exact replica
of Groq's internal rate-limit implementation.
"""

import asyncio

from collections import deque
from dataclasses import dataclass
from time import monotonic
from uuid import uuid4


@dataclass(frozen=True)
class AdmissionLimits:
    rpm: int = 30
    tpm: int = 8000
    rpd: int = 1000
    tpd: int = 200000

    max_in_flight: int = 8
    max_queue: int = 20

    minute_window_s: float = 60.0
    day_window_s: float = 86400.0

    def __post_init__(self):

        for name in (
            "rpm",
            "tpm",
            "rpd",
            "tpd",
            "max_in_flight",
            "max_queue",
        ):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be >= 1"
                )

        if (
            self.minute_window_s <= 0
            or self.day_window_s <= 0
        ):
            raise ValueError(
                "Window durations must be positive"
            )


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    reason: str

    queue_wait_ms: float
    reserved_tokens: int

    queue_depth_at_arrival: int
    queue_peak: int

    in_flight_at_admission: int = 0


class AdmissionController:

    def __init__(self, limits: AdmissionLimits):

        self.limits = limits

        self._condition = asyncio.Condition()
        self._pending = deque()

        now = monotonic()

        self._minute_start = now
        self._day_start = now

        self._minute_requests = 0
        self._minute_tokens = 0

        self._day_requests = 0
        self._day_tokens = 0

        self._in_flight = 0

        self.peak_queue = 0
        self.peak_in_flight = 0

    def _refresh(self, now):

        if (
            now - self._minute_start
            >= self.limits.minute_window_s
        ):
            self._minute_start = now

            self._minute_requests = 0
            self._minute_tokens = 0

        if (
            now - self._day_start
            >= self.limits.day_window_s
        ):
            self._day_start = now

            self._day_requests = 0
            self._day_tokens = 0

    def _daily_exhausted(self, tokens):

        return (
            self._day_requests >= self.limits.rpd
            or
            self._day_tokens + tokens > self.limits.tpd
        )

    def _minute_blocked(self, tokens):

        return (
            self._minute_requests >= self.limits.rpm
            or
            self._minute_tokens + tokens > self.limits.tpm
        )

    async def acquire(
        self,
        estimated_tokens: int,
        max_wait_s: float = 2.0,
    ):

        if estimated_tokens <= 0:
            raise ValueError(
                "estimated_tokens must be positive"
            )

        if max_wait_s < 0:
            raise ValueError(
                "max_wait_s cannot be negative"
            )

        start = monotonic()
        deadline = start + max_wait_s

        ticket = uuid4().hex

        async with self._condition:

            self._refresh(monotonic())

            depth = len(self._pending)

            def denied(reason):

                return AdmissionDecision(
                    accepted=False,
                    reason=reason,
                    queue_wait_ms=round(
                        (monotonic() - start) * 1000,
                        2,
                    ),
                    reserved_tokens=0,
                    queue_depth_at_arrival=depth,
                    queue_peak=self.peak_queue,
                )

            # Reject requests too large for one window.
            if (
                estimated_tokens > self.limits.tpm
                or
                estimated_tokens > self.limits.tpd
            ):
                return denied(
                    "request_exceeds_capacity"
                )

            # Daily budget exhaustion is not retried.
            if self._daily_exhausted(estimated_tokens):
                return denied("daily_exhausted")

            # Bounded queue to prevent unlimited backlog.
            if len(self._pending) >= self.limits.max_queue:
                return denied("queue_full")

            self._pending.append(ticket)

            try:

                while True:

                    now = monotonic()

                    self._refresh(now)

                    if self._daily_exhausted(
                        estimated_tokens
                    ):
                        return denied(
                            "daily_exhausted"
                        )

                    is_first = (
                        self._pending[0] == ticket
                    )

                    minute_limited = (
                        self._minute_blocked(
                            estimated_tokens
                        )
                    )

                    concurrency_busy = (
                        self._in_flight
                        >= self.limits.max_in_flight
                    )

                    # FIFO: only the oldest waiting
                    # request can acquire capacity.
                    if (
                        is_first
                        and not minute_limited
                        and not concurrency_busy
                    ):

                        self._pending.popleft()

                        self._minute_requests += 1
                        self._day_requests += 1

                        self._minute_tokens += (
                            estimated_tokens
                        )

                        self._day_tokens += (
                            estimated_tokens
                        )

                        self._in_flight += 1

                        self.peak_in_flight = max(
                            self.peak_in_flight,
                            self._in_flight,
                        )

                        self._condition.notify_all()

                        return AdmissionDecision(
                            accepted=True,
                            reason="accepted",
                            queue_wait_ms=round(
                                (monotonic() - start)
                                * 1000,
                                2,
                            ),
                            reserved_tokens=(
                                estimated_tokens
                            ),
                            queue_depth_at_arrival=depth,
                            queue_peak=self.peak_queue,
                            in_flight_at_admission=(
                                self._in_flight
                            ),
                        )

                    # Request must wait.
                    self.peak_queue = max(
                        self.peak_queue,
                        len(self._pending),
                    )

                    remaining = deadline - now

                    if remaining <= 0:
                        return denied(
                            "queue_timeout"
                        )

                    # Wake on capacity release or when
                    # the minute window resets.
                    if minute_limited:

                        next_reset = (
                            self._minute_start
                            + self.limits.minute_window_s
                            - now
                        )

                        remaining = min(
                            remaining,
                            max(0.001, next_reset),
                        )

                    try:

                        await asyncio.wait_for(
                            self._condition.wait(),
                            timeout=remaining,
                        )

                    except asyncio.TimeoutError:
                        pass

            finally:

                # Remove abandoned or rejected tickets.
                if ticket in self._pending:

                    self._pending.remove(ticket)
                    self._condition.notify_all()

    async def release(self):

        async with self._condition:

            if self._in_flight <= 0:
                raise RuntimeError(
                    "release() called without "
                    "an active lease"
                )

            self._in_flight -= 1

            self._condition.notify_all()

    async def snapshot(self):

        async with self._condition:

            self._refresh(monotonic())

            return {
                "minute_requests": (
                    self._minute_requests
                ),
                "minute_tokens_reserved": (
                    self._minute_tokens
                ),
                "day_requests": (
                    self._day_requests
                ),
                "day_tokens_reserved": (
                    self._day_tokens
                ),
                "queue_length": len(self._pending),
                "peak_queue": self.peak_queue,
                "in_flight": self._in_flight,
                "peak_in_flight": (
                    self.peak_in_flight
                ),
            }
