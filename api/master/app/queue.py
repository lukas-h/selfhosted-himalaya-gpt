from __future__ import annotations

import asyncio
import secrets
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def new_job_id() -> str:
    return "job_" + secrets.token_urlsafe(12)


@dataclass
class Job:
    id: str
    model: str
    request: dict[str, Any]
    deadline_at: float
    out_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=64))
    done: asyncio.Event = field(default_factory=asyncio.Event)
    created_at: float = field(default_factory=time.time)
    assigned_worker: str | None = None
    assigned_at: float | None = None
    semaphore: asyncio.Semaphore | None = None
    semaphore_released: bool = False
    user_disconnected: bool = False
    finalized: bool = False
    finalized_at: float | None = None
    final_reason: str | None = None
    # True when the request asked for tools: master injected the Hermes prompt
    # and must parse <tool_call> spans out of the model output (see streaming.py).
    tool_mode: bool = False


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
        self.last_poll_by_model: dict[tuple[str, str], float] = {}
        self.finalized_counts: Counter[str] = Counter()
        self.drop_counts: Counter[str] = Counter()
        self.late_stream_count = 0

    def semaphore_for(self, worker_id: str) -> asyncio.Semaphore:
        sem = self._sems.get(worker_id)
        if sem is None:
            sem = asyncio.Semaphore(self._sem_cap)
            self._sems[worker_id] = sem
        return sem

    def register(self, job: Job) -> None:
        self.in_flight[job.id] = job

    def pop(self, job_id: str) -> Job | None:
        return self.finalize(job_id, "popped")

    def get(self, job_id: str) -> Job | None:
        return self.in_flight.get(job_id)

    def mark_poll(self, worker_id: str, model: str | None = None) -> None:
        now = time.time()
        self.last_poll_at[worker_id] = now
        if model is not None:
            self.last_poll_by_model[(worker_id, model)] = now

    def any_recent_poll(self, max_age_s: float) -> bool:
        now = time.time()
        return any(now - t < max_age_s for t in self.last_poll_at.values())

    def skip_reason(self, job: Job, now: float | None = None) -> str | None:
        now = time.time() if now is None else now
        if job.id not in self.in_flight:
            return "stale"
        if job.user_disconnected:
            return "client_disconnected"
        if now >= job.deadline_at:
            return "expired"
        return None

    def finalize(self, job_id: str, reason: str) -> Job | None:
        job = self.in_flight.pop(job_id, None)
        if job is None:
            return None

        if not job.finalized:
            job.finalized = True
            job.final_reason = reason
            job.finalized_at = time.time()
            self.finalized_counts[reason] += 1
        if job.semaphore is not None and not job.semaphore_released:
            job.semaphore.release()
            job.semaphore_released = True
        job.done.set()
        return job

    def prune_queue(self, model: str, now: float | None = None) -> int:
        queue = self.queues[model]
        now = time.time() if now is None else now
        dropped = 0
        keep: list[Job] = []

        for _ in range(queue.qsize()):
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            reason = self.skip_reason(job, now)
            if reason is None:
                keep.append(job)
                continue
            self.drop_counts[reason] += 1
            self.finalize(job.id, reason)
            dropped += 1

        for job in keep:
            queue.put_nowait(job)
        return dropped

    def note_late_stream(self) -> None:
        self.late_stream_count += 1

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        queues: dict[str, dict[str, Any]] = {}
        for slug, queue in self.queues.items():
            queued_jobs = list(getattr(queue, "_queue", []))
            oldest = min((job.created_at for job in queued_jobs), default=None)
            queues[slug] = {
                "depth": queue.qsize(),
                "oldest_age_s": round(now - oldest, 3) if oldest is not None else None,
            }

        active = []
        for job in self.in_flight.values():
            active.append({
                "id": job.id,
                "model": job.model,
                "age_s": round(now - job.created_at, 3),
                "deadline_in_s": round(job.deadline_at - now, 3),
                "assigned_worker": job.assigned_worker,
                "assigned_age_s": (
                    round(now - job.assigned_at, 3) if job.assigned_at is not None else None
                ),
                "user_disconnected": job.user_disconnected,
            })

        return {
            "queues": queues,
            "in_flight": active,
            "last_poll_by_worker": {
                worker_id: round(now - ts, 3) for worker_id, ts in self.last_poll_at.items()
            },
            "last_poll_by_model": {
                f"{worker_id}:{model}": round(now - ts, 3)
                for (worker_id, model), ts in self.last_poll_by_model.items()
            },
            "finalized_counts": dict(self.finalized_counts),
            "drop_counts": dict(self.drop_counts),
            "late_stream_count": self.late_stream_count,
            "worker_capacity": self._sem_cap,
        }
