"""Unit tests for the disconnect watcher in master/app/routes/openai.py.

The watcher's job is to set `job.user_disconnected = True` as quickly as
possible after the client drops, so the puller can skip the phantom job
when it dequeues. We test:

1. Idle client → flag stays False, watcher exits cleanly when stopped.
2. Client disconnects → flag flips True within the first 100 ms tick.
3. Watcher stops promptly when its `stop` event is set (the parent
   handler returning).
4. `request.is_disconnected()` raising → watcher exits gracefully without
   leaking the flag.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

# Make api/master importable without polluting the test runner's sys.path
MASTER = Path(__file__).resolve().parents[1] / "master"
sys.path.insert(0, str(MASTER))
from app.queue import Job  # noqa: E402
from app.routes import openai as openai_route  # noqa: E402


class FakeRequest:
    """Just enough of starlette.Request for `_watch_disconnect`."""

    def __init__(self, *, disconnect_after: float = float("inf"), raise_after: float | None = None):
        self._loop_start = None
        self._disconnect_after = disconnect_after
        self._raise_after = raise_after

    async def is_disconnected(self) -> bool:
        if self._loop_start is None:
            self._loop_start = asyncio.get_event_loop().time()
        elapsed = asyncio.get_event_loop().time() - self._loop_start
        if self._raise_after is not None and elapsed >= self._raise_after:
            raise RuntimeError("simulated receive failure")
        return elapsed >= self._disconnect_after


def _new_job() -> Job:
    return Job(id="job_test", model="himalaya-q8", request={})


@pytest.mark.asyncio
async def test_watcher_does_nothing_when_client_stays_connected() -> None:
    job = _new_job()
    stop = asyncio.Event()
    req = FakeRequest()  # never disconnects

    task = asyncio.create_task(openai_route._watch_disconnect(req, job, stop))
    await asyncio.sleep(0.3)
    assert not job.user_disconnected
    stop.set()
    await asyncio.wait_for(task, timeout=0.5)


@pytest.mark.asyncio
async def test_watcher_flips_flag_within_one_tick() -> None:
    job = _new_job()
    stop = asyncio.Event()
    req = FakeRequest(disconnect_after=0.05)  # disconnect after 50 ms

    task = asyncio.create_task(openai_route._watch_disconnect(req, job, stop))
    # First tick is immediate (poll before sleep), second ~100 ms later.
    # Either way the flag should be True by 250 ms.
    await asyncio.sleep(0.25)
    assert job.user_disconnected, "watcher should have detected disconnect by 250 ms"
    await asyncio.wait_for(task, timeout=0.5)


@pytest.mark.asyncio
async def test_watcher_stops_promptly_on_stop_event() -> None:
    job = _new_job()
    stop = asyncio.Event()
    req = FakeRequest()

    task = asyncio.create_task(openai_route._watch_disconnect(req, job, stop))
    # Fire stop right away, watcher should drain in well under one tick.
    stop.set()
    await asyncio.wait_for(task, timeout=0.2)
    assert not job.user_disconnected


@pytest.mark.asyncio
async def test_watcher_swallows_receive_errors() -> None:
    """If `request.is_disconnected()` raises (which we've seen happen in
    practice when the receive channel is in a bad state), the watcher
    must exit without flipping the flag — the chat handler's own timeout
    will clean up."""
    job = _new_job()
    stop = asyncio.Event()
    req = FakeRequest(raise_after=0.0)

    task = asyncio.create_task(openai_route._watch_disconnect(req, job, stop))
    await asyncio.wait_for(task, timeout=0.3)
    assert not job.user_disconnected
