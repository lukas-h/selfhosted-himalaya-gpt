"""Long-polling puller. One coroutine per slug, sharing a global concurrency semaphore."""
from __future__ import annotations

import asyncio
import logging
import random

import httpx

from .runner import run_job

log = logging.getLogger(__name__)


async def puller(
    slug: str,
    sem: asyncio.Semaphore,
    master: httpx.AsyncClient,
    llama: httpx.AsyncClient,
    worker_id: str,
    poll_timeout_s: int,
) -> None:
    backoff = 1.0
    while True:
        await sem.acquire()  # hold a slot before polling, release on job completion
        slot_handed_off = False
        try:
            try:
                resp = await master.get(
                    "/internal/jobs/poll",
                    params={"model": slug, "worker_id": worker_id, "timeout": poll_timeout_s},
                    timeout=poll_timeout_s + 15,
                )
            except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                log.warning("[%s] poll error: %r (sleep %.1fs)", slug, exc, backoff)
                await asyncio.sleep(backoff + random.random())
                backoff = min(backoff * 2, 30.0)
                continue

            backoff = 1.0
            if resp.status_code == 204:
                continue
            if resp.status_code >= 400:
                log.warning("[%s] poll http %s: %s", slug, resp.status_code, resp.text[:200])
                await asyncio.sleep(2.0)
                continue

            job = resp.json()

            async def _wrap():
                try:
                    await run_job(job, master, llama)
                finally:
                    sem.release()

            asyncio.create_task(_wrap())
            slot_handed_off = True
        finally:
            if not slot_handed_off:
                sem.release()
