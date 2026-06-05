"""End-to-end HTTP integration test for tool calling.

Unlike test_tool_calling.py (which exercises the formatter/parser in isolation)
this drives the REAL master ASGI app over HTTP: a client POSTs
`/v1/chat/completions` with `tools`, a simulated worker long-polls
`/internal/jobs/poll` and streams `<tool_call>` chunks back via
`/internal/jobs/{id}/stream`, and we assert the client gets OpenAI-shaped
`tool_calls`. This covers the glue the unit tests don't: the request handler's
`apply_tool_formatting` (tools → Hermes prompt, tools stripped, `Job.tool_mode`
set) wired to the streaming/aggregate parse.

Auth reads module-level settings, so we bypass it with dependency_overrides and
inject a test Settings/Registry into app.state.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

MASTER = Path(__file__).resolve().parents[1] / "master"
sys.path.insert(0, str(MASTER))

from app.auth import require_user_token, require_worker_token  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.queue import Registry  # noqa: E402

SLUG = "himalaya-test"
USER_HDR = {"Authorization": "Bearer testkey"}


@pytest.fixture
def app():
    a = create_app()
    # bypass token auth (it reads module settings); drive routes off app.state
    a.dependency_overrides[require_user_token] = lambda: "u"
    a.dependency_overrides[require_worker_token] = lambda: "w"
    a.state.settings = Settings(model_slugs=SLUG, user_request_timeout_s=20, worker_poll_timeout_s=10)
    a.state.registry = Registry([SLUG], max_queue_depth=32, max_concurrent_per_worker=3)
    return a


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _envelope(obj) -> bytes:
    return json.dumps(obj).encode() + b"\n"


async def _fake_worker(client, content: str, *, finish: str = "stop", split: bool = True):
    """Long-poll for the job, then stream `content` back as NDJSON chunk
    envelopes (split across two deltas to mimic token streaming) + done.
    Returns the polled job dict so tests can assert the request the worker saw."""
    job = None
    for _ in range(60):
        r = await client.get("/internal/jobs/poll",
                             params={"model": SLUG, "worker_id": "w1", "timeout": 2})
        if r.status_code == 200:
            job = r.json()
            break
        await asyncio.sleep(0.02)
    assert job is not None, "worker never received a job"

    async def body():
        pieces = ([content[:len(content) // 2], content[len(content) // 2:]] if split else [content])
        for p in pieces:
            yield _envelope({"type": "chunk", "data": {"choices": [{"delta": {"content": p}, "finish_reason": None}]}})
        yield _envelope({"type": "chunk", "data": {"choices": [{"delta": {}, "finish_reason": finish}]}})
        yield _envelope({"type": "done"})

    await client.post(f"/internal/jobs/{job['job_id']}/stream", content=body(),
                      headers={"Content-Type": "application/x-ndjson"})
    return job


def _tools():
    return [{"type": "function", "function": {
        "name": "get_weather", "description": "Get weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


async def test_e2e_non_streaming_tool_call(app):
    """tools in → the worker sees a Hermes-formatted, tools-stripped request →
    a <tool_call> reply comes back as OpenAI tool_calls + finish_reason."""
    tool_text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Kathmandu"}}</tool_call>'
    req = {"model": SLUG, "messages": [{"role": "user", "content": "weather in KTM?"}],
           "tools": _tools(), "stream": False}
    async with _client(app) as client:
        client_task = asyncio.create_task(client.post("/v1/chat/completions", json=req, headers=USER_HDR))
        job = await _fake_worker(client, tool_text)
        resp = await client_task

    assert resp.status_code == 200, resp.text
    data = resp.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["message"]["tool_calls"][0]
    assert tc["type"] == "function" and tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Kathmandu"}
    assert choice["message"]["content"] in (None, "")

    # the worker must have received the request with tools stripped + a Hermes prompt injected
    sent = job["request"]
    assert "tools" not in sent and "tool_choice" not in sent
    assert any("<tools>" in (m.get("content") or "") for m in sent["messages"]), \
        "Hermes <tools> prompt was not injected into the downstream request"


async def test_e2e_streaming_tool_call(app):
    """Streaming tool turn: buffered, then a single tool_calls delta + [DONE]."""
    tool_text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Pokhara"}}</tool_call>'
    req = {"model": SLUG, "messages": [{"role": "user", "content": "weather?"}],
           "tools": _tools(), "stream": True}
    payloads = []
    async with _client(app) as client:
        worker_task = asyncio.create_task(_fake_worker(client, tool_text))
        async with client.stream("POST", "/v1/chat/completions", json=req, headers=USER_HDR) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    payloads.append(line[6:])
        await worker_task

    assert "[DONE]" in payloads
    deltas = [json.loads(p) for p in payloads if p != "[DONE]"]
    tool_chunks = [d for d in deltas if d.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
    assert tool_chunks, f"no tool_calls delta in stream: {payloads}"
    choice = tool_chunks[-1]["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["delta"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    assert tc["index"] == 0  # required by the OpenAI streaming schema


async def test_e2e_no_tools_is_plain_chat(app):
    """No tools → unchanged passthrough: plain content, no tool_calls, worker
    sees the original messages (no Hermes injection)."""
    req = {"model": SLUG, "messages": [{"role": "user", "content": "hi"}], "stream": False}
    async with _client(app) as client:
        client_task = asyncio.create_task(client.post("/v1/chat/completions", json=req, headers=USER_HDR))
        job = await _fake_worker(client, "Hello there!")
        resp = await client_task

    data = resp.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "Hello there!"
    assert "tool_calls" not in choice["message"]
    # untouched request: no Hermes prompt, single user message
    assert job["request"]["messages"] == [{"role": "user", "content": "hi"}]
