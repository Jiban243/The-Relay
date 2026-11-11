
import asyncio

from src.admission import (
    AdmissionController,
    AdmissionLimits,
)


def test_admit_release():

    async def run():

        ctrl = AdmissionController(
            AdmissionLimits()
        )

        lease = await ctrl.acquire(500)

        assert lease.accepted

        assert (
            await ctrl.snapshot()
        )["in_flight"] == 1

        await ctrl.release()

        assert (
            await ctrl.snapshot()
        )["in_flight"] == 0

    asyncio.run(run())


def test_daily_budget_exhausted():

    async def run():

        ctrl = AdmissionController(
            AdmissionLimits(rpd=1)
        )

        assert (
            await ctrl.acquire(100)
        ).accepted

        await ctrl.release()

        rejected = await ctrl.acquire(100)

        assert not rejected.accepted
        assert rejected.reason == "daily_exhausted"

    asyncio.run(run())


def test_minute_limit_recovers():

    async def run():

        ctrl = AdmissionController(
            AdmissionLimits(
                rpm=1,
                minute_window_s=0.05,
            )
        )

        assert (
            await ctrl.acquire(100)
        ).accepted

        await ctrl.release()

        lease = await ctrl.acquire(
            100,
            max_wait_s=0.25,
        )

        assert lease.accepted
        assert lease.queue_wait_ms >= 10

        await ctrl.release()

    asyncio.run(run())


def test_bounded_queue_wait():

    async def run():

        ctrl = AdmissionController(
            AdmissionLimits(
                max_in_flight=1
            )
        )

        assert (
            await ctrl.acquire(100)
        ).accepted

        rejected = await ctrl.acquire(
            100,
            max_wait_s=0.01,
        )

        assert rejected.reason == "queue_timeout"

        assert (
            await ctrl.snapshot()
        )["peak_queue"] >= 1

        await ctrl.release()

    asyncio.run(run())


def test_queue_full_and_fifo():

    async def run():

        ctrl = AdmissionController(
            AdmissionLimits(
                max_in_flight=1,
                max_queue=1,
            )
        )

        assert (
            await ctrl.acquire(100)
        ).accepted

        waiting = asyncio.create_task(
            ctrl.acquire(
                100,
                max_wait_s=0.3,
            )
        )

        await asyncio.sleep(0.02)

        rejected = await ctrl.acquire(
            100,
            max_wait_s=0.01,
        )

        assert rejected.reason == "queue_full"

        await ctrl.release()

        lease = await waiting

        assert lease.accepted

        await ctrl.release()

    asyncio.run(run())


def test_request_too_large():

    async def run():

        ctrl = AdmissionController(
            AdmissionLimits(tpm=100)
        )

        rejected = await ctrl.acquire(101)

        assert not rejected.accepted

        assert (
            rejected.reason
            == "request_exceeds_capacity"
        )

    asyncio.run(run())
