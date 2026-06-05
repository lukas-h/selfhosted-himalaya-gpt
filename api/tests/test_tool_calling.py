"""Unit tests for Hermes function-calling support (app.tool_calling) and its
wiring into the master response shaping (app.streaming).

These are fast and model-free: they pin the prompt FORMATTING (Hermes preamble +
<tools> block + synthesized few-shot — the few-shot is what actually makes the
0.5B model emit tool calls, see test_tool_calling_eval.py for the live proof)
and the output PARSING (<tool_call> primary, <|python_*|>/fenced/bare-JSON
fallbacks), plus the OpenAI tool_calls shaping.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

MASTER = Path(__file__).resolve().parents[1] / "master"
sys.path.insert(0, str(MASTER))

from app.queue import Job  # noqa: E402
from app.streaming import (  # noqa: E402
    _assistant_message,
    _tool_stream_chunks,
    aggregate_chat_completion,
)
from app.tool_calling import (  # noqa: E402
    apply_tool_formatting,
    build_tool_system_prompt,
    parse_tool_calls,
    render_messages_with_tools,
    synthesize_fewshot,
    to_openai_tool_calls,
    wants_tools,
)


def _weather_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def _search_tool():
    return {
        "type": "function",
        "function": {
            "name": "internet_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


# -----------------------------------------------------------------------------
# Request-side: apply_tool_formatting / wants_tools
# -----------------------------------------------------------------------------

def test_apply_tool_formatting_rewrites_and_strips():
    req = {
        "messages": [{"role": "user", "content": "weather in KTM?"}],
        "tools": [_weather_tool()],
        "tool_choice": "auto",
    }
    assert apply_tool_formatting(req) is True
    # tool fields removed so the worker/llama-server never see them
    for k in ("tools", "tool_choice", "functions", "function_call"):
        assert k not in req
    roles = [m["role"] for m in req["messages"]]
    # system, then a synthesized few-shot user/assistant pair, then the real user
    assert roles[0] == "system"
    assert roles[-1] == "user"
    assert req["messages"][-1]["content"] == "weather in KTM?"
    assert "assistant" in roles  # few-shot exemplar present


def test_tool_choice_none_disables_but_strips():
    req = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [_weather_tool()],
        "tool_choice": "none",
    }
    assert apply_tool_formatting(req) is False
    assert "tools" not in req and "tool_choice" not in req
    # messages untouched (no Hermes prompt injected)
    assert req["messages"] == [{"role": "user", "content": "hi"}]


def test_no_tools_is_passthrough():
    req = {"messages": [{"role": "user", "content": "hi"}]}
    assert apply_tool_formatting(req) is False
    assert req["messages"] == [{"role": "user", "content": "hi"}]


def test_legacy_functions_field():
    req = {
        "messages": [{"role": "user", "content": "search cats"}],
        "functions": [_search_tool()["function"]],
    }
    assert wants_tools(req) is True
    assert apply_tool_formatting(req) is True
    assert "functions" not in req
    assert any("internet_search" in (m.get("content") or "") for m in req["messages"])


def test_empty_tools_list_not_tool_mode():
    req = {"messages": [{"role": "user", "content": "hi"}], "tools": []}
    assert apply_tool_formatting(req) is False


# -----------------------------------------------------------------------------
# System prompt + few-shot synthesis
# -----------------------------------------------------------------------------

def test_system_prompt_contains_tools_and_format():
    sp = build_tool_system_prompt([_weather_tool()["function"]])
    assert "<tools>" in sp and "</tools>" in sp
    assert "get_weather" in sp
    assert "<tool_call>" in sp  # output format instruction present


def test_system_prompt_required_and_forced_directives():
    fns = [_weather_tool()["function"]]
    assert "must call at least one tool" in build_tool_system_prompt(fns, "required").lower()
    forced = build_tool_system_prompt(fns, {"type": "function", "function": {"name": "get_weather"}})
    assert "get_weather" in forced and "must call" in forced.lower()


def test_fewshot_synthesis_shapes_valid_tool_calls():
    fns = [_weather_tool()["function"], _search_tool()["function"]]
    shots = synthesize_fewshot(fns, n=2)
    assert len(shots) == 4  # two user/assistant pairs
    assistants = [m for m in shots if m["role"] == "assistant"]
    assert len(assistants) == 2
    for a, fn in zip(assistants, fns):
        clean, calls = parse_tool_calls(a["content"])
        assert len(calls) == 1
        assert calls[0]["name"] == fn["name"]
        # example args cover the required params
        assert set(calls[0]["arguments"]) == set(fn["parameters"]["required"])


def test_fewshot_count_capped_by_tools():
    shots = synthesize_fewshot([_weather_tool()["function"]], n=2)
    assert len(shots) == 2  # only one tool → one pair


# -----------------------------------------------------------------------------
# History rendering (tool results, prior assistant tool calls)
# -----------------------------------------------------------------------------

def test_render_tool_role_becomes_tool_response():
    msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "KTM"}'}}]},
        {"role": "tool", "name": "get_weather", "tool_call_id": "call_0", "content": "22C sunny"},
        {"role": "user", "content": "thanks"},
    ]
    out = render_messages_with_tools(msgs, [_weather_tool()["function"]], n_fewshot=0)
    text = "\n".join(m.get("content") or "" for m in out)
    assert "<tool_response>" in text and "22C sunny" in text
    # prior assistant tool call rendered back to <tool_call> text
    assert "<tool_call>" in text and "get_weather" in text


# -----------------------------------------------------------------------------
# Output parsing
# -----------------------------------------------------------------------------

def test_parse_single_call_strips_span():
    clean, calls = parse_tool_calls(
        'Sure. <tool_call>{"name": "get_weather", "arguments": {"city": "KTM"}}</tool_call>')
    assert calls == [{"name": "get_weather", "arguments": {"city": "KTM"}}]
    assert clean == "Sure."


def test_parse_parallel_calls():
    clean, calls = parse_tool_calls(
        '<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>'
        '<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>')
    assert [c["name"] for c in calls] == ["a", "b"]


def test_parse_preserves_non_ascii_arguments():
    raw = '<tool_call>{"name": "internet_search", "arguments": {"query": "काठमाडौंको मौसम"}}</tool_call>'
    _, calls = parse_tool_calls(raw)
    assert calls[0]["arguments"]["query"] == "काठमाडौंको मौसम"


def test_parse_unclosed_tool_call():
    # length-truncated output with no closing tag
    _, calls = parse_tool_calls('<tool_call>{"name":"get_weather","arguments":{"city":"KTM"}}')
    assert calls == [{"name": "get_weather", "arguments": {"city": "KTM"}}]


def test_parse_python_token_json_and_call():
    assert parse_tool_calls('<|python_start|>{"name":"f","arguments":{"a":1}}<|python_end|>')[1] \
        == [{"name": "f", "arguments": {"a": 1}}]
    assert parse_tool_calls('<|python_start|>get_weather(city="KTM")<|python_end|>')[1] \
        == [{"name": "get_weather", "arguments": {"city": "KTM"}}]


def test_parse_fenced_json_fallback():
    text = 'Here you go:\n```json\n{"name": "get_weather", "arguments": {"city": "KTM"}}\n```'
    _, calls = parse_tool_calls(text)
    assert calls == [{"name": "get_weather", "arguments": {"city": "KTM"}}]


def test_parse_bare_json_object_fallback():
    _, calls = parse_tool_calls('{"name": "get_weather", "arguments": {"city": "KTM"}}')
    assert calls == [{"name": "get_weather", "arguments": {"city": "KTM"}}]


def test_parse_arguments_given_as_json_string():
    _, calls = parse_tool_calls('<tool_call>{"name": "f", "arguments": "{\\"a\\": 1}"}</tool_call>')
    assert calls == [{"name": "f", "arguments": {"a": 1}}]


def test_parse_malformed_is_skipped():
    clean, calls = parse_tool_calls("<tool_call>{not valid json}</tool_call>")
    assert calls == []


def test_parse_plain_text_passthrough():
    clean, calls = parse_tool_calls("The capital of Nepal is Kathmandu.")
    assert calls == [] and clean == "The capital of Nepal is Kathmandu."


def test_parse_empty():
    assert parse_tool_calls("") == ("", [])


# -----------------------------------------------------------------------------
# OpenAI tool_calls shaping
# -----------------------------------------------------------------------------

def test_to_openai_tool_calls_shape_and_deterministic_ids():
    out = to_openai_tool_calls([
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {"y": "z"}},
    ])
    assert [c["id"] for c in out] == ["call_0", "call_1"]
    assert all(c["type"] == "function" for c in out)
    # arguments must be a JSON string per the OpenAI schema
    assert out[0]["function"]["arguments"] == '{"x": 1}'
    assert json.loads(out[1]["function"]["arguments"]) == {"y": "z"}


# -----------------------------------------------------------------------------
# Master response shaping (app.streaming integration)
# -----------------------------------------------------------------------------

def _job(tool_mode: bool) -> Job:
    return Job(id="job_abc", model="himalaya-q8", request={},
               deadline_at=time.time() + 60, tool_mode=tool_mode)


def _chunk(content="", finish=None):
    return {"choices": [{"delta": {"content": content} if content else {}, "finish_reason": finish}]}


def test_aggregate_emits_tool_calls_in_tool_mode():
    job = _job(tool_mode=True)
    chunks = [
        _chunk('<tool_call>{"name": "get_weather", '),
        _chunk('"arguments": {"city": "KTM"}}</tool_call>', finish="stop"),
    ]
    resp = aggregate_chat_completion(job, chunks, "himalaya-q8")
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tcs = choice["message"]["tool_calls"]
    assert tcs[0]["function"]["name"] == "get_weather"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"city": "KTM"}
    assert choice["message"]["content"] is None


def test_aggregate_plain_text_when_tool_mode_but_no_call():
    job = _job(tool_mode=True)
    chunks = [_chunk("Kathmandu is the capital.", finish="stop")]
    resp = aggregate_chat_completion(job, chunks, "himalaya-q8")
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "Kathmandu is the capital."
    assert "tool_calls" not in choice["message"]


def test_aggregate_non_tool_mode_is_unchanged():
    job = _job(tool_mode=False)
    chunks = [_chunk("hi", finish="stop")]
    msg = aggregate_chat_completion(job, chunks, "himalaya-q8")["choices"][0]["message"]
    assert msg == {"role": "assistant", "content": "hi"}


def test_assistant_message_helper():
    job = _job(tool_mode=True)
    msg, fr = _assistant_message(job, '<tool_call>{"name":"f","arguments":{}}</tool_call>', None)
    assert fr == "tool_calls" and msg["tool_calls"][0]["function"]["name"] == "f"


def test_tool_stream_chunks_emit_tool_calls_delta():
    job = _job(tool_mode=True)
    chunks = _tool_stream_chunks(
        job, '<tool_call>{"name":"get_weather","arguments":{"city":"KTM"}}</tool_call>', None)
    assert len(chunks) == 1
    choice = chunks[0]["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["delta"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    assert tc["index"] == 0  # OpenAI streaming requires index on each tool_call delta
    assert chunks[0]["object"] == "chat.completion.chunk"


def test_tool_stream_chunks_fallback_to_text():
    job = _job(tool_mode=True)
    chunks = _tool_stream_chunks(job, "just text", "stop")
    choice = chunks[0]["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["delta"]["content"] == "just text"
