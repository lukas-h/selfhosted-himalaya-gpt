"""Unit tests for master-side queue lifecycle and stale-job cleanup."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

MASTER = Path(__file__).resolve().parents[1] / "master"
sys.path.insert(0, str(MASTER))
from app.queue import Job, Registry  # noqa: E402
from app.routes import internal  # noqa: E402
from app.streaming import sse_for  # noqa: E402


def _job(job_id: str, *, deadline_at: float | None = None) -> Job:
    return Job(
        id=job_id,
        model="himalaya-q8",
        request={"model": "himalaya-q8"},
        deadline_at=deadline_at if deadline_at is not None else time.time() + 60,
    )


def _request(registry: Registry):
    settings = SimpleNamespace(
        slugs=["himalaya-q8"],
        worker_poll_timeout_s=1,
        user_request_timeout_s=30,
        max_concurrent_per_worker=1,
        max_queue_depth=8,
        rate_limit_rpm=1200,
        rate_limit_burst_rps=30,
    )
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        settings=settings,
        registry=registry,
    )))


class ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


def test_finalize_releases_worker_semaphore_once() -> None:
    registry = Registry(["himalaya-q8"], max_queue_depth=8, max_concurrent_per_worker=1)
    job = _job("job_release")
    sem = registry.semaphore_for("worker-a")
    assert sem._value == 1

    async def _acquire():
        await sem.acquire()

    import asyncio

    asyncio.run(_acquire())
    job.semaphore = sem
    registry.register(job)
    assert sem._value == 0

    assert registry.finalize(job.id, "user_timeout") is job
    assert sem._value == 1
    assert registry.finalize(job.id, "late_worker_done") is None
    assert sem._value == 1


@pytest.mark.asyncio
async def test_poll_skips_expired_job_and_assigns_live_job() -> None:
    registry = Registry(["himalaya-q8"], max_queue_depth=8, max_concurrent_per_worker=1)
    expired = _job("job_expired", deadline_at=time.time() - 1)
    live = _job("job_live", deadline_at=time.time() + 60)
    registry.register(expired)
    registry.register(live)
    registry.queues["himalaya-q8"].put_nowait(expired)
    registry.queues["himalaya-q8"].put_nowait(live)

    result = await internal.poll(
        _request(registry),
        model="himalaya-q8",
        worker_id="worker-a",
        timeout=1,
        _="token",
    )

    assert isinstance(result, dict)
    assert result["job_id"] == live.id
    assert registry.get(expired.id) is None
    assert registry.finalized_counts["expired"] == 1
    assert registry.drop_counts["expired"] == 1
    assert registry.semaphore_for("worker-a")._value == 0

    registry.finalize(live.id, "test_done")
    assert registry.semaphore_for("worker-a")._value == 1


@pytest.mark.asyncio
async def test_late_worker_stream_returns_gone_without_capacity_leak() -> None:
    registry = Registry(["himalaya-q8"], max_queue_depth=8, max_concurrent_per_worker=1)

    with pytest.raises(HTTPException) as exc:
        await internal.stream("job_missing", _request(registry), _="token")

    assert exc.value.status_code == 410
    assert registry.late_stream_count == 1


@pytest.mark.asyncio
async def test_sse_for_enforces_job_deadline() -> None:
    job = _job("job_stream_timeout", deadline_at=time.time() - 1)

    chunks = [chunk async for chunk in sse_for(job, ConnectedRequest())]

    payload = b"".join(chunks)
    assert b"timeout_error" in payload
    assert payload.endswith(b"data: [DONE]\n\n")
