"""SSE encoding (master → user) and NDJSON parsing (worker → master).

Worker uploads a chunked-body POST whose payload is one JSON envelope per line:

    {"type": "chunk",  "data": <openai delta>}
    {"type": "done"}
    {"type": "error", "data": {"message": "...", "type": "..."}}

Master forwards `chunk.data` as SSE `data:` events to the user, ending with
the OpenAI sentinel `data: [DONE]`.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import orjson
from fastapi import Request

from .queue import Job

KEEPALIVE_INTERVAL_S = 15.0


async def parse_ndjson(request: Request, on_envelope) -> None:
    """Read a chunked request body, decode NDJSON envelopes, await `on_envelope` each."""
    buf = bytearray()
    async for chunk in request.stream():
        if not chunk:
            continue
        buf.extend(chunk)
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(buf[:nl]).strip()
            del buf[: nl + 1]
            if not line:
                continue
            try:
                env = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            await on_envelope(env)
    # tail (no trailing newline)
    line = bytes(buf).strip()
    if line:
        try:
            env = orjson.loads(line)
            await on_envelope(env)
        except orjson.JSONDecodeError:
            pass


async def sse_for(job: Job, request: Request) -> AsyncIterator[bytes]:
    """SSE generator for the user-facing /v1/chat/completions stream."""
    while True:
        if await request.is_disconnected():
            job.user_disconnected = True
            return
        remaining = job.deadline_at - time.time()
        if remaining <= 0:
            err = {"message": "upstream timeout", "type": "timeout_error"}
            yield b"data: " + orjson.dumps({"error": err}) + b"\n\n"
            yield b"data: [DONE]\n\n"
            return
        try:
            envelope = await asyncio.wait_for(
                job.out_queue.get(),
                timeout=min(KEEPALIVE_INTERVAL_S, remaining),
            )
        except asyncio.TimeoutError:
            yield b": keepalive\n\n"
            continue

        kind = envelope.get("type")
        if kind == "chunk":
            yield b"data: " + orjson.dumps(envelope.get("data", {})) + b"\n\n"
        elif kind == "done":
            yield b"data: [DONE]\n\n"
            return
        elif kind == "error":
            err = envelope.get("data", {"message": "unknown error", "type": "server_error"})
            yield b"data: " + orjson.dumps({"error": err}) + b"\n\n"
            yield b"data: [DONE]\n\n"
            return
        else:
            # ignore unknown envelope types (forward-compat)
            continue


def aggregate_chat_completion(job: Job, chunks: list[dict], model: str) -> dict:
    """Build a single OpenAI chat.completion object from streamed deltas."""
    content_parts: list[str] = []
    finish_reason = None
    for ch in chunks:
        choices = ch.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if isinstance(delta.get("content"), str):
            content_parts.append(delta["content"])
        if choices[0].get("finish_reason"):
            finish_reason = choices[0]["finish_reason"]

    full_text = "".join(content_parts)
    return {
        "id": "chatcmpl-" + job.id.split("_", 1)[-1],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": finish_reason or "stop",
            }
        ],
        "usage": None,
    }
