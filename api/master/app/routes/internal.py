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

    registry.mark_poll(worker_id)
    sem = registry.semaphore_for(worker_id)

    # acquire a slot before pulling a job; release happens when /stream completes
    try:
        await asyncio.wait_for(sem.acquire(), timeout=timeout)
    except asyncio.TimeoutError:
        return Response(status_code=204)

    try:
        job = await asyncio.wait_for(registry.queues[model].get(), timeout=timeout)
    except asyncio.TimeoutError:
        sem.release()
        return Response(status_code=204)

    job.assigned_worker = worker_id
    job.semaphore = sem
    return {
        "job_id": job.id,
        "model": job.model,
        "request": job.request,
        "issued_at": time.time(),
        "deadline_at": job.created_at + settings.user_request_timeout_s,
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
        raise HTTPException(status_code=404, detail="job not found")

    final_marker_seen = {"seen": False}

    async def on_envelope(env: dict) -> None:
        # always feed the user's stream
        try:
            await asyncio.wait_for(job.out_queue.put(env), timeout=10.0)
        except asyncio.TimeoutError:
            # user is too slow (or gone); drop instead of blocking the upload
            pass
        if env.get("type") in ("done", "error"):
            final_marker_seen["seen"] = True

    try:
        await parse_ndjson(request, on_envelope)
    except Exception as exc:  # disconnect, parse error, etc.
        await job.out_queue.put({
            "type": "error",
            "data": {"message": f"worker upload failed: {exc!r}", "type": "server_error"},
        })
    finally:
        if not final_marker_seen["seen"]:
            await job.out_queue.put({
                "type": "error",
                "data": {"message": "worker disconnected before completion", "type": "server_error"},
            })
        if job.semaphore is not None:
            job.semaphore.release()
        job.done.set()

    return {"ok": True}
