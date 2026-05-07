"""Run a single job: stream from local llama-server, push NDJSON envelopes back to master."""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import httpx
import orjson

from . import llama_client

log = logging.getLogger(__name__)


async def _envelopes(job: dict, llama: httpx.AsyncClient) -> AsyncIterator[bytes]:
    """Async generator yielding NDJSON-framed envelope lines (with trailing \\n)."""
    body = dict(job["request"])
    body["model"] = job["model"]
    body["stream"] = True
    sent_terminal = False
    try:
        async for ev in llama_client.stream_chat(llama, body):
            if isinstance(ev, str) and ev == "[DONE]":
                yield orjson.dumps({"type": "done"}) + b"\n"
                sent_terminal = True
                return
            if isinstance(ev, dict) and "_error" in ev:
                yield orjson.dumps({
                    "type": "error",
                    "data": {"message": ev["_error"], "type": "upstream_error"},
                }) + b"\n"
                sent_terminal = True
                return
            yield orjson.dumps({"type": "chunk", "data": ev}) + b"\n"
    except (httpx.HTTPError, asyncio.CancelledError) as exc:
        yield orjson.dumps({
            "type": "error",
            "data": {"message": f"upstream stream failed: {exc!r}", "type": "upstream_error"},
        }) + b"\n"
        sent_terminal = True
    finally:
        if not sent_terminal:
            yield orjson.dumps({
                "type": "error",
                "data": {"message": "stream ended unexpectedly", "type": "upstream_error"},
            }) + b"\n"


async def run_job(
    job: dict,
    master: httpx.AsyncClient,
    llama: httpx.AsyncClient,
) -> None:
    job_id = job["job_id"]
    try:
        # Chunked-body POST: httpx infers Transfer-Encoding: chunked when content is async iter.
        resp = await master.post(
            f"/internal/jobs/{job_id}/stream",
            content=_envelopes(job, llama),
            headers={"Content-Type": "application/x-ndjson"},
            timeout=httpx.Timeout(connect=10, write=None, read=60, pool=None),
        )
        if resp.status_code >= 400:
            log.warning("master rejected job %s upload: %s %s", job_id, resp.status_code, resp.text[:200])
    except Exception as exc:
        log.warning("job %s upload failed: %r", job_id, exc)
