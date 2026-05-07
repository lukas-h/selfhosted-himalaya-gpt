from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Any


def new_job_id() -> str:
    return "job_" + secrets.token_urlsafe(12)


@dataclass
class Job:
    id: str
    model: str
    request: dict[str, Any]
    out_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=64))
    done: asyncio.Event = field(default_factory=asyncio.Event)
    created_at: float = field(default_factory=time.time)
    assigned_worker: str | None = None
    semaphore: asyncio.Semaphore | None = None
    user_disconnected: bool = False


class Registry:
    """Per-slug job queues + in-flight registry + per-worker concurrency.

    Lives as part of FastAPI app.state — see main.py.
    """

    def __init__(self, slugs: list[str], max_queue_depth: int, max_concurrent_per_worker: int):
        self.queues: dict[str, asyncio.Queue[Job]] = {
            slug: asyncio.Queue(maxsize=max_queue_depth) for slug in slugs
        }
        self.in_flight: dict[str, Job] = {}
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._sem_cap = max_concurrent_per_worker
        self.last_poll_at: dict[str, float] = {}

    def semaphore_for(self, worker_id: str) -> asyncio.Semaphore:
        sem = self._sems.get(worker_id)
        if sem is None:
            sem = asyncio.Semaphore(self._sem_cap)
            self._sems[worker_id] = sem
        return sem

    def register(self, job: Job) -> None:
        self.in_flight[job.id] = job

    def pop(self, job_id: str) -> Job | None:
        return self.in_flight.pop(job_id, None)

    def get(self, job_id: str) -> Job | None:
        return self.in_flight.get(job_id)

    def mark_poll(self, worker_id: str) -> None:
        self.last_poll_at[worker_id] = time.time()

    def any_recent_poll(self, max_age_s: float) -> bool:
        now = time.time()
        return any(now - t < max_age_s for t in self.last_poll_at.values())
