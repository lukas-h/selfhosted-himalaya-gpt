from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from ..auth import require_worker_token
from ..streaming import parse_ndjson

router = APIRouter(prefix="/internal")


@router.get("/jobs/poll")
async def poll(
    request: Request,
    model: str = Query(...),
    worker_id: str = Query(...),
    timeout: int = Query(default=55, ge=1, le=120),
    _: str = Depends(require_worker_token),
):
    settings = request.app.state.settings
    registry = request.app.state.registry
    timeout = min(timeout, settings.worker_poll_timeout_s)

    if model not in settings.slugs:
        raise HTTPException(status_code=400, detail=f"unknown model '{model}'")

    registry.mark_poll(worker_id, model)
    sem = registry.semaphore_for(worker_id)

    deadline = time.time() + timeout
    while True:
        registry.prune_queue(model)
        remaining = max(0.1, deadline - time.time())
        try:
            job = await asyncio.wait_for(registry.queues[model].get(), timeout=remaining)
        except asyncio.TimeoutError:
            return Response(status_code=204)
        reason = registry.skip_reason(job)
        if reason is not None:
            registry.drop_counts[reason] += 1
            registry.finalize(job.id, reason)
            continue

        try:
            await asyncio.wait_for(sem.acquire(), timeout=0.25)
        except asyncio.TimeoutError:
            if registry.skip_reason(job) is None:
                try:
                    registry.queues[model].put_nowait(job)
                except asyncio.QueueFull:
                    registry.finalize(job.id, "capacity_requeue_failed")
            return Response(status_code=204)

        reason = registry.skip_reason(job)
        if reason is not None:
            sem.release()
            registry.drop_counts[reason] += 1
            registry.finalize(job.id, reason)
            continue
        break

    job.assigned_worker = worker_id
    job.assigned_at = time.time()
    job.semaphore = sem
    job.semaphore_released = False
    return {
        "job_id": job.id,
        "model": job.model,
        "request": job.request,
        "issued_at": time.time(),
        "deadline_at": job.deadline_at,
    }


@router.post("/jobs/{job_id}/stream")
async def stream(
    job_id: str,
    request: Request,
    _: str = Depends(require_worker_token),
):
    registry = request.app.state.registry
    job = registry.get(job_id)
    if job is None:
        registry.note_late_stream()
        raise HTTPException(status_code=410, detail="job no longer active")

    terminal_reason = {"reason": None}

    async def on_envelope(env: dict) -> None:
        if job.finalized:
            return
        # always feed the user's stream
        try:
            await asyncio.wait_for(job.out_queue.put(env), timeout=10.0)
        except asyncio.TimeoutError:
            # user is too slow (or gone); drop instead of blocking the upload
            pass
        if env.get("type") == "done":
            terminal_reason["reason"] = "worker_done"
        elif env.get("type") == "error":
            terminal_reason["reason"] = "worker_error"

    try:
        await parse_ndjson(request, on_envelope)
    except Exception as exc:  # disconnect, parse error, etc.
        terminal_reason["reason"] = "worker_upload_failed"
        if not job.finalized:
            await job.out_queue.put({
                "type": "error",
                "data": {"message": f"worker upload failed: {exc!r}", "type": "server_error"},
            })
    finally:
        if terminal_reason["reason"] is None:
            terminal_reason["reason"] = "worker_disconnected"
            if not job.finalized:
                await job.out_queue.put({
                    "type": "error",
                    "data": {
                        "message": "worker disconnected before completion",
                        "type": "server_error",
                    },
                })
        registry.finalize(job.id, terminal_reason["reason"])

    return {"ok": True}


@router.get("/status")
async def status(
    request: Request,
    _: str = Depends(require_worker_token),
):
    settings = request.app.state.settings
    registry = request.app.state.registry
    return {
        "settings": {
            "slugs": settings.slugs,
            "max_concurrent_per_worker": settings.max_concurrent_per_worker,
            "max_queue_depth": settings.max_queue_depth,
            "user_request_timeout_s": settings.user_request_timeout_s,
            "worker_poll_timeout_s": settings.worker_poll_timeout_s,
            "rate_limit_rpm": settings.rate_limit_rpm,
            "rate_limit_burst_rps": settings.rate_limit_burst_rps,
        },
        "registry": registry.snapshot(),
    }
