"""Validate the master's CONSTRUCTED responses against the real OpenAI pydantic
models, so spec drift fails a test instead of a client at runtime.

This is the test that would have caught the missing streaming tool_call `index`
(see test_validation_has_teeth below): hand-asserting the fields we *expect*
green-lit a response the OpenAI SDK rejects. Validating against the SDK's own
`ChatCompletion` / `ChatCompletionChunk` models checks the WHOLE shape — required
fields, types, nested tool_call deltas — automatically.

Only the master-constructed outputs are checked here: `aggregate_chat_completion`
(non-streaming) and `_tool_stream_chunks` (the buffered tool stream). A
representative llama-server passthrough chunk is checked too, since the master
forwards those verbatim to the client.

Skipped if `openai` isn't installed (dev dependency).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

MASTER = Path(__file__).resolve().parents[1] / "master"
sys.path.insert(0, str(MASTER))

pytest.importorskip("openai")
from openai.types.chat import ChatCompletion, ChatCompletionChunk  # noqa: E402

from app.queue import Job  # noqa: E402
from app.streaming import _tool_stream_chunks, aggregate_chat_completion  # noqa: E402


def _job(tool_mode: bool) -> Job:
    return Job(id="job_abc", model="himalaya-q8", request={},
               deadline_at=time.time() + 60, tool_mode=tool_mode)


def _delta_chunk(content="", finish=None):
    return {"choices": [{"delta": {"content": content} if content else {}, "finish_reason": finish}]}


# -----------------------------------------------------------------------------
# Non-streaming: aggregate_chat_completion -> ChatCompletion
# -----------------------------------------------------------------------------

def test_plain_completion_matches_chatcompletion():
    out = aggregate_chat_completion(_job(False), [_delta_chunk("hello", "stop")], "himalaya-q8")
    cc = ChatCompletion.model_validate(out)  # raises on any spec violation
    assert cc.choices[0].message.content == "hello"
    assert cc.choices[0].finish_reason == "stop"


def test_tool_completion_matches_chatcompletion():
    chunks = [_delta_chunk('<tool_call>{"name":"get_weather",'),
              _delta_chunk('"arguments":{"city":"KTM"}}</tool_call>', "stop")]
    out = aggregate_chat_completion(_job(True), chunks, "himalaya-q8")
    cc = ChatCompletion.model_validate(out)
    assert cc.choices[0].finish_reason == "tool_calls"
    assert cc.choices[0].message.tool_calls[0].function.name == "get_weather"


def test_parallel_tool_completion_matches_chatcompletion():
    text = ('<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>'
            '<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>')
    out = aggregate_chat_completion(_job(True), [_delta_chunk(text, "stop")], "himalaya-q8")
    cc = ChatCompletion.model_validate(out)
    assert [t.function.name for t in cc.choices[0].message.tool_calls] == ["a", "b"]


# -----------------------------------------------------------------------------
# Streaming: _tool_stream_chunks -> ChatCompletionChunk
# -----------------------------------------------------------------------------

def test_tool_stream_chunk_matches_chunk_schema():
    chunks = _tool_stream_chunks(
        _job(True), '<tool_call>{"name":"get_weather","arguments":{"city":"KTM"}}</tool_call>', None)
    assert chunks
    for ch in chunks:
        cc = ChatCompletionChunk.model_validate(ch)  # would FAIL without tool_call index
        tc = cc.choices[0].delta.tool_calls[0]
        assert tc.index == 0
        assert tc.function.name == "get_weather"


def test_parallel_tool_stream_chunk_matches_chunk_schema():
    text = ('<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>'
            '<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>')
    for ch in _tool_stream_chunks(_job(True), text, None):
        cc = ChatCompletionChunk.model_validate(ch)
        assert [t.index for t in cc.choices[0].delta.tool_calls] == [0, 1]


def test_tool_stream_text_fallback_matches_chunk_schema():
    # tool_mode but the model produced no call -> buffered text as a content delta
    for ch in _tool_stream_chunks(_job(True), "just prose, no call", "stop"):
        ChatCompletionChunk.model_validate(ch)


# -----------------------------------------------------------------------------
# Passthrough: a representative llama-server chunk the master forwards verbatim
# (shape captured from the live deployment)
# -----------------------------------------------------------------------------

def test_forwarded_llama_chunk_matches_chunk_schema():
    forwarded = {
        "id": "chatcmpl-abc", "object": "chat.completion.chunk", "created": 1,
        "model": "himalaya-q8", "system_fingerprint": "b0-unknown",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}],
    }
    ChatCompletionChunk.model_validate(forwarded)


# -----------------------------------------------------------------------------
# Meta: prove the schema check actually has teeth
# -----------------------------------------------------------------------------

def test_validation_has_teeth_missing_tool_index():
    """A streaming tool_call delta WITHOUT `index` — the exact bug we shipped —
    must fail ChatCompletionChunk validation. If this ever passes, the schema
    check is toothless."""
    bad = {
        "id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {
            "role": "assistant",
            "tool_calls": [{"id": "call_0", "type": "function",
                            "function": {"name": "f", "arguments": "{}"}}],  # no index
        }}],
    }
    with pytest.raises(Exception):
        ChatCompletionChunk.model_validate(bad)
