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
from .tool_calling import parse_tool_calls, strip_leaked_markup, to_openai_tool_calls

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
    """SSE generator for the user-facing /v1/chat/completions stream.

    Normal requests pass deltas straight through. Tool requests (job.tool_mode)
    buffer the model output and emit a single `tool_calls` delta at the end — a
    `<tool_call>` span can't be recognized until it's fully received, so token
    streaming is traded for correct tool parsing on tool turns only.
    """
    buffering = job.tool_mode
    buf: list[str] = []
    finish_reason: str | None = None
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
            data = envelope.get("data", {})
            if buffering:
                _accumulate(buf, data)
                fr = _chunk_finish_reason(data)
                if fr:
                    finish_reason = fr
                continue
            yield b"data: " + orjson.dumps(data) + b"\n\n"
        elif kind == "done":
            if buffering:
                for chunk in _tool_stream_chunks(job, "".join(buf), finish_reason):
                    yield b"data: " + orjson.dumps(chunk) + b"\n\n"
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
    message, finish_reason = _assistant_message(job, full_text, finish_reason)
    return {
        "id": "chatcmpl-" + job.id.split("_", 1)[-1],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": None,
    }


def _assistant_message(job: Job, full_text: str, finish_reason: str | None) -> tuple[dict, str]:
    """Shape the assistant message. In tool_mode, parse <tool_call> spans into an
    OpenAI tool_calls array (finish_reason → "tool_calls"); otherwise plain text."""
    if job.tool_mode:
        clean, calls = parse_tool_calls(full_text)
        clean = strip_leaked_markup(clean)  # drop any echoed <tool_response>/<tool_call>
        if calls:
            return (
                {"role": "assistant", "content": clean or None,
                 "tool_calls": to_openai_tool_calls(calls)},
                "tool_calls",
            )
        return {"role": "assistant", "content": clean}, (finish_reason or "stop")
    return {"role": "assistant", "content": full_text}, (finish_reason or "stop")


def _accumulate(buf: list[str], data: dict) -> None:
    choices = data.get("choices") or []
    if choices:
        delta = choices[0].get("delta") or {}
        if isinstance(delta.get("content"), str):
            buf.append(delta["content"])


def _chunk_finish_reason(data: dict) -> str | None:
    choices = data.get("choices") or []
    if choices and choices[0].get("finish_reason"):
        return choices[0]["finish_reason"]
    return None


def _tool_stream_chunks(job: Job, full_text: str, finish_reason: str | None) -> list[dict]:
    """Final streaming chunk(s) for a buffered tool turn: one tool_calls delta if
    a call was parsed, else the buffered text as a single content delta."""
    clean, calls = parse_tool_calls(full_text)
    clean = strip_leaked_markup(clean)  # drop any echoed <tool_response>/<tool_call>
    base = {
        "id": "chatcmpl-" + job.id.split("_", 1)[-1],
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": job.model,
    }
    if calls:
        tool_calls = to_openai_tool_calls(calls)
        # OpenAI streaming requires an `index` on each tool_call delta (the SDK
        # uses it to assemble fragmented calls and marks it required). Without it
        # strict clients raise a validation error mid-stream.
        for idx, tc in enumerate(tool_calls):
            tc["index"] = idx
        delta: dict = {"role": "assistant", "tool_calls": tool_calls}
        if clean:
            delta["content"] = clean
        return [{**base, "choices": [{"index": 0, "delta": delta, "finish_reason": "tool_calls"}]}]
    return [{**base, "choices": [
        {"index": 0, "delta": {"role": "assistant", "content": clean},
         "finish_reason": finish_reason or "stop"}]}]
