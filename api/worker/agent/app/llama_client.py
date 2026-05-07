"""Thin async wrapper that streams SSE chunks from local llama-server.

We don't use httpx-sse because llama-server's SSE framing is simple enough
to parse with `aiter_lines`, and skipping the dep keeps the worker leaner.
"""
from __future__ import annotations

from typing import AsyncIterator

import httpx
import orjson


async def stream_chat(
    client: httpx.AsyncClient,
    body: dict,
) -> AsyncIterator[dict | str]:
    """Yields each upstream SSE event payload.

    Yields:
      - dict for normal `data: {json}` events
      - the literal string "[DONE]" when the stream ends naturally
      - dict {"_error": "..."} on HTTP error from llama-server (worker still wraps it)
    """
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={**body, "stream": True},
        timeout=httpx.Timeout(connect=10, read=None, write=30, pool=None),
    ) as resp:
        if resp.status_code != 200:
            text = (await resp.aread()).decode(errors="replace")
            yield {"_error": f"llama-server {resp.status_code}: {text[:300]}"}
            return
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].lstrip()
            if payload == "[DONE]":
                yield "[DONE]"
                return
            try:
                yield orjson.loads(payload)
            except orjson.JSONDecodeError:
                continue
