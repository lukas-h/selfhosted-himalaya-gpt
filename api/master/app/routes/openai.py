from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import require_user_token
from ..config import settings
from ..queue import Job, new_job_id
from ..ratelimit import limiter
from ..schemas import ChatCompletionRequest, ModelObject, ModelsList
from ..streaming import aggregate_chat_completion, sse_for

router = APIRouter(prefix="/v1")


@router.get("/models", response_model=ModelsList)
async def list_models(request: Request, _: str = Depends(require_user_token)) -> ModelsList:
    return ModelsList(data=[ModelObject(id=slug) for slug in request.app.state.settings.slugs])


async def _watch_disconnect(request: Request, job: Job, stop: asyncio.Event) -> None:
    """Mark job.user_disconnected when the client drops, so the poll handler
    can skip phantom jobs instead of routing them to a worker.

    Polls every 100 ms — fast enough that the puller's poll handler (which
    only dequeues when its long-poll wakes) almost always sees the flag
    before pulling the job. Drops to a slow 1 s heartbeat after the first
    minute since most disconnects happen early.
    """
    polled = 0
    while not stop.is_set():
        try:
            if await request.is_disconnected():
                job.user_disconnected = True
                stop.set()
                return
        except Exception:
            return
        # First minute: tight 100 ms cadence. After that: 1 s — long-running
        # generations don't need millisecond-precise disconnect detection.
        interval = 0.1 if polled < 600 else 1.0
        polled += 1
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


@router.post("/chat/completions")
@limiter.limit(f"{settings.rate_limit_burst_rps}/second;{settings.rate_limit_rpm}/minute"
               if settings.rate_limit_burst_rps > 0
               else f"{settings.rate_limit_rpm}/minute")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    _: str = Depends(require_user_token),
):
    cfg = request.app.state.settings
    registry = request.app.state.registry

    if body.model not in cfg.slugs:
        raise HTTPException(status_code=400, detail=f"unknown model '{body.model}'")

    job = Job(
        id=new_job_id(),
        model=body.model,
        request=body.model_dump(exclude_none=True),
    )
    registry.register(job)

    # Spawn the disconnect watcher BEFORE we put the job on the worker queue.
    # Otherwise a fast puller can dequeue and dispatch the job before the
    # watcher's first tick sets user_disconnected, wasting worker cycles on
    # an already-abandoned client.
    stop = asyncio.Event()
    watcher = asyncio.create_task(_watch_disconnect(request, job, stop))

    try:
        await asyncio.wait_for(registry.queues[body.model].put(job), timeout=5.0)
    except (asyncio.TimeoutError, asyncio.QueueFull):
        stop.set()
        watcher.cancel()
        registry.pop(job.id)
        raise HTTPException(status_code=503, detail="queue full, try again")

    if body.stream:
        # The job lifetime extends past this function's return until the SSE
        # generator finishes. Pop happens inside the generator's finally.
        async def _stream():
            try:
                async for chunk in sse_for(job, request):
                    yield chunk
            finally:
                stop.set()
                watcher.cancel()
                registry.pop(job.id)

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    try:
        chunks: list[dict] = []
        deadline = time.time() + cfg.user_request_timeout_s
        while True:
            timeout = max(1.0, deadline - time.time())
            try:
                envelope = await asyncio.wait_for(job.out_queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="upstream timeout")
            if job.user_disconnected:
                # Client gave up while waiting; bail without raising so we
                # don't pollute logs. Job stays flagged for the puller to
                # skip if it hasn't been picked up yet.
                return JSONResponse(status_code=499, content={"error": {
                    "message": "client disconnected", "type": "client_closed",
                }})
            kind = envelope.get("type")
            if kind == "chunk":
                chunks.append(envelope.get("data") or {})
            elif kind == "done":
                break
            elif kind == "error":
                err = envelope.get("data", {"message": "upstream error", "type": "server_error"})
                return JSONResponse(status_code=502, content={"error": err})
        return JSONResponse(content=aggregate_chat_completion(job, chunks, body.model))
    finally:
        stop.set()
        watcher.cancel()
        try:
            await watcher
        except (asyncio.CancelledError, Exception):
            pass
        registry.pop(job.id)
