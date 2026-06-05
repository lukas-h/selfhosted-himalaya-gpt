"""Shared helpers for the function-calling evals (smoke + BFCL-style).

These talk DIRECTLY to a llama-server so the eval exercises the real model plus
the master's Hermes formatter/parser (app.tool_calling) without needing the full
master -> worker stack. Point it at a running server:

    HIMALAYA_EVAL_LLAMA_URL=http://127.0.0.1:9990

Optionally HIMALAYA_EVAL_LLAMA_KEY if the server was started with LLAMA_API_KEY.

The same render_messages_with_tools / parse_tool_calls used in production are
used here, so the eval measures exactly what a client would get.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

MASTER = Path(__file__).resolve().parents[1] / "master"
sys.path.insert(0, str(MASTER))

from app.tool_calling import parse_tool_calls, render_messages_with_tools  # noqa: E402

LLAMA_URL = os.environ.get("HIMALAYA_EVAL_LLAMA_URL", "")
API_KEY = os.environ.get("HIMALAYA_EVAL_LLAMA_KEY", "")
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tool_calling"


def have_server() -> bool:
    return bool(LLAMA_URL)


def _functions(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in tools:
        out.append(t["function"] if t.get("type") == "function" and "function" in t else t)
    return out


def llama_chat(messages: list[dict[str, Any]], *, max_tokens: int = 220,
               temperature: float = 0.2, base: str | None = None) -> str:
    """One non-streaming chat completion against the llama-server."""
    base = base or LLAMA_URL
    body = {
        "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
        "top_k": 40, "repeat_penalty": 1.08, "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"].get("content") or ""


def run_tool_case(tools: list[dict[str, Any]], user_messages: list[dict[str, Any]],
                  *, max_tokens: int = 220, base: str | None = None
                  ) -> tuple[str, list[dict[str, Any]]]:
    """Render tools+messages exactly as the master does, query the model, and
    parse tool calls back out. Returns (raw_text, calls)."""
    msgs = render_messages_with_tools(user_messages, _functions(tools))
    text = llama_chat(msgs, max_tokens=max_tokens, base=base)
    _clean, calls = parse_tool_calls(text)
    return text, calls


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for f in sorted(FIXTURES.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        cases.extend(data if isinstance(data, list) else [data])
    return cases


def tool_names(case: dict[str, Any]) -> set[str]:
    return {(t.get("function") or t)["name"] for t in case["tools"]}
