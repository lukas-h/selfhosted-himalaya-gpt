"""Tiny mock of llama-server's OpenAI-style endpoints, just enough for plumbing tests.

Streams 12 fake "chunks" at 80ms intervals on /v1/chat/completions.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

API_KEY = os.environ.get("LLAMA_API_KEY", "")
SLUGS = os.environ.get("MOCK_SLUGS", "himalaya-bf16,himalaya-q8,himalaya-q4").split(",")

app = FastAPI()


def _check_auth(authorization: str | None) -> None:
    if not API_KEY:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401)
    if authorization.split(" ", 1)[1].strip() != API_KEY:
        raise HTTPException(401)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    return {"object": "list", "data": [{"id": s, "object": "model", "owned_by": "mock"} for s in SLUGS]}


@app.post("/v1/chat/completions")
async def chat(req: Request, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    body = await req.json()
    model = body.get("model", "unknown")
    stream = bool(body.get("stream"))
    msg = body.get("messages", [{}])[-1].get("content", "")
    response_text = f"[mock {model}] echo: {msg!r}".replace("\n", " ")
    pieces = [response_text[i : i + max(1, len(response_text) // 12)] for i in range(0, len(response_text), max(1, len(response_text) // 12))]
    cmpl_id = "chatcmpl-" + uuid.uuid4().hex[:12]

    async def gen():
        for i, p in enumerate(pieces):
            yield "data: " + json.dumps({
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": p} if i == 0 else {"content": p},
                    "finish_reason": None,
                }],
            }) + "\n\n"
            await asyncio.sleep(0.08)
        yield "data: " + json.dumps({
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }) + "\n\n"
        yield "data: [DONE]\n\n"

    if stream:
        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse({
        "id": cmpl_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
    })


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
