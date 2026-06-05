"""Hermes-style function calling for the nanochat himalayagpt model.

Why this module exists
----------------------
The OpenAI `tools` field used to pass straight through master → worker →
llama-server. The nanochat GGUF chat template has no tool support ("Tool/
developer roles render as user content"), so the model never saw a real tool
schema and just hallucinated or looped on the raw instruction text — the
reported bug (the Nepali `get_nepal_live_context …` system prompt echoed with
`---` separators).

What the model actually does (verified by probing the prebuilt fork server):

  * It does NOT emit nanochat `<|python_*|>` special tokens, and it does NOT
    emit `<tool_call>` zero-shot from just a system prompt — it answers in prose
    or loops.
  * Given a Hermes system prompt PLUS one or two few-shot exemplars, it reliably
    emits well-formed `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
    and stops cleanly. Few-shot is the trigger; without it the 0.5B model ignores
    the format instruction.

So this module, entirely in the master (the worker stays a dumb forwarder):

  1. `apply_tool_formatting` rewrites the request `messages` into a Hermes prompt
     (preamble + `<tools>` block + synthesized few-shot exemplars) and removes the
     `tools`/`tool_choice`/`functions` fields so llama-server never sees them.
  2. `parse_tool_calls` extracts tool calls back out of the model's text
     (primary: `<tool_call>`; defensive: `<|python_*|>`, fenced/bare JSON) and
     `to_openai_tool_calls` shapes them into the OpenAI `tool_calls` array.

Tool *selection* accuracy is a property of the 0.5B model, not this module — the
eval suite measures it. This module's job is correct formatting and parsing.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any

# How many synthesized few-shot exemplars to inject. 2 is enough to lock the
# format in (1 already works; 2 adds a little robustness for tool selection).
DEFAULT_FEWSHOT = 2

# The canonical NousResearch/Hermes function-calling preamble, adapted with an
# explicit "output only the <tool_call> line, do not write code" instruction —
# the himalayagpt SFT otherwise likes to answer with a markdown ```python block.
_PREAMBLE = (
    "You are a function calling AI model. You are provided with function "
    "signatures within <tools></tools> XML tags. You may call one or more "
    "functions to assist with the user query. Don't make assumptions about what "
    "values to plug into functions. Here are the available tools:"
)
_FORMAT_INSTRUCTION = (
    "For each function call, return a json object with the function name and "
    "arguments within <tool_call></tool_call> XML tags, like:\n"
    '<tool_call>{"name": <function-name>, "arguments": <args-dict>}</tool_call>\n'
    "Output only the <tool_call> line(s) when a tool is needed — do not explain "
    "and do not write code. If no tool is needed, answer normally."
)

# Tags / fences we parse out of model output.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>\s*(.*)\Z", re.DOTALL)  # unclosed (length-truncated)
_PYTHON_TOKEN_RE = re.compile(r"<\|python_start\|>\s*(.*?)\s*<\|python_end\|>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json|tool_call|python)?\s*(.*?)\s*```", re.DOTALL)


# ---------------------------------------------------------------------------
# Request side: format tools + few-shot into messages
# ---------------------------------------------------------------------------

def _functions_from_request(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize OpenAI `tools` (and legacy `functions`) to a list of function
    objects {name, description, parameters}."""
    fns: list[dict[str, Any]] = []
    for t in request.get("tools") or []:
        if isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("function"), dict):
            fns.append(t["function"])
        elif isinstance(t, dict) and "name" in t:  # already a bare function obj
            fns.append(t)
    for f in request.get("functions") or []:  # legacy OpenAI functions=
        if isinstance(f, dict) and "name" in f:
            fns.append(f)
    return [f for f in fns if f.get("name")]


# Name-keyword → sample value for the few-shot exemplar. Only UNAMBIGUOUS names
# get a concrete value; ambiguous short names (e.g. "to"/"from", which mean
# recipient in send_email but currency in convert) are intentionally absent so
# they fall through to a clear `<slot>` placeholder rather than a misleading
# constant. The real lesson is taught by echoing whatever value we pick in the
# exemplar's USER turn (see synthesize_fewshot), steering the model to EXTRACT
# from the query rather than copy.
_SAMPLE_KEYWORDS: list[tuple[tuple[str, ...], Any]] = [
    (("city", "town"), "Paris"),
    (("location", "place"), "Paris"),
    (("country", "nation"), "Nepal"),
    (("url", "link", "uri", "href", "webpage"), "https://example.com/article"),
    (("symbol", "ticker"), "AAPL"),
    (("recipient", "mailto"), "alex@example.com"),
    (("date", "day"), "2026-01-01"),
]


def _example_value(pname: str, schema: dict[str, Any]) -> Any:
    """Pick a sample value for a JSON-schema property for the few-shot exemplar.
    Unambiguous names get a realistic value; everything else gets a `<slot>`
    placeholder so the model treats it as fill-in (and won't copy a constant)."""
    if not isinstance(schema, dict):
        schema = {}
    if "example" in schema:
        return schema["example"]
    if "default" in schema:
        return schema["default"]
    if schema.get("enum"):
        return schema["enum"][0]
    t = schema.get("type", "string")
    low = (pname or "").lower()
    if t == "string":
        for keys, val in _SAMPLE_KEYWORDS:
            if any(k in low for k in keys):
                return val
        return f"<{pname or 'value'}>"
    return {"integer": 1, "number": 1, "boolean": True, "array": [], "object": {}}.get(t, f"<{pname or 'value'}>")


def _example_args(fn: dict[str, Any]) -> dict[str, Any]:
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    required = params.get("required") or list(props.keys())
    return {k: _example_value(k, props.get(k, {})) for k in required}


def build_tool_system_prompt(functions: list[dict[str, Any]], tool_choice: Any = None) -> str:
    """Hermes system prompt: preamble + <tools> block + output-format rules."""
    tools_block = "\n".join(json.dumps(fn, ensure_ascii=False) for fn in functions)
    parts = [_PREAMBLE, f"<tools>\n{tools_block}\n</tools>", _FORMAT_INSTRUCTION]
    forced = _forced_tool_name(tool_choice)
    if tool_choice == "required":
        parts.append("You must call at least one tool for this query.")
    elif forced:
        parts.append(f"You must call the `{forced}` tool for this query.")
    return "\n".join(parts)


def synthesize_fewshot(functions: list[dict[str, Any]], n: int = DEFAULT_FEWSHOT) -> list[dict[str, Any]]:
    """Build n synthetic user/assistant exemplar turns from the real tools.

    Two things make this work on the 0.5B model (both verified by probing):
      * the format must be DEMONSTRATED, not just described — a bare system
        prompt is ignored, one exemplar already triggers `<tool_call>` output;
      * the exemplar's USER turn must contain the argument values, so the model
        learns to EXTRACT values from the query instead of copying the example's
        placeholder verbatim into the real call.
    """
    msgs: list[dict[str, Any]] = []
    for fn in functions[:max(0, n)]:
        args = _example_args(fn)
        if args:
            shown = "; ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
            user = f"(example) Use `{fn['name']}` with {shown}."
        else:
            user = f"(example) Use `{fn['name']}`."
        call = {"name": fn["name"], "arguments": args}
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant",
                     "content": f"<tool_call>{json.dumps(call, ensure_ascii=False)}</tool_call>"})
    return msgs


def _render_history_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Map an OpenAI message to a nanochat-renderable (system/user/assistant)
    message, turning tool calls/results into Hermes text. Returns None to drop."""
    role = msg.get("role")
    if role == "tool":
        # Tool result → Hermes <tool_response> in a user turn (nanochat has no tool role).
        name = msg.get("name") or ""
        content = msg.get("content")
        payload = {"name": name, "content": content}
        return {"role": "user",
                "content": f"<tool_response>{json.dumps(payload, ensure_ascii=False)}</tool_response>"}
    if role == "assistant" and msg.get("tool_calls"):
        # Prior assistant tool call → render back to <tool_call> text.
        rendered = []
        for tc in msg["tool_calls"]:
            fn = (tc or {}).get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            rendered.append(f"<tool_call>{json.dumps({'name': fn.get('name'), 'arguments': args}, ensure_ascii=False)}</tool_call>")
        text = (msg.get("content") or "")
        return {"role": "assistant", "content": (text + "\n" + "\n".join(rendered)).strip()}
    return msg


def render_messages_with_tools(
    messages: list[dict[str, Any]],
    functions: list[dict[str, Any]],
    tool_choice: Any = None,
    n_fewshot: int = DEFAULT_FEWSHOT,
) -> list[dict[str, Any]]:
    """Produce the message list to send downstream: tool system prompt, then
    synthesized few-shot exemplars, then the (tool-rendered) real conversation."""
    out: list[dict[str, Any]] = [
        {"role": "system", "content": build_tool_system_prompt(functions, tool_choice)}
    ]
    out.extend(synthesize_fewshot(functions, n_fewshot))
    for msg in messages:
        rendered = _render_history_message(msg)
        if rendered is not None:
            out.append(rendered)
    return out


def _forced_tool_name(tool_choice: Any) -> str | None:
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if isinstance(fn, dict):
            return fn.get("name")
    return None


def wants_tools(request: dict[str, Any]) -> bool:
    """True if the request asks for tool use (and hasn't disabled it)."""
    if request.get("tool_choice") == "none":
        return False
    return bool(request.get("tools") or request.get("functions"))


def apply_tool_formatting(request: dict[str, Any]) -> bool:
    """Mutate `request` in place for tool calling and return tool_mode.

    When tools are requested: rewrite `messages` into the Hermes prompt and pop
    the tool fields so the worker/llama-server never receive them. When tools are
    present but disabled (`tool_choice: "none"`) we still strip the fields (the
    nanochat server can't use them) but return False (no tool parsing).
    """
    tool_mode = wants_tools(request)
    if tool_mode:
        functions = _functions_from_request(request)
        if functions:
            request["messages"] = render_messages_with_tools(
                request.get("messages") or [], functions, request.get("tool_choice"))
        else:
            tool_mode = False
    # Always remove tool fields from the downstream request.
    for k in ("tools", "tool_choice", "functions", "function_call"):
        request.pop(k, None)
    return tool_mode


# ---------------------------------------------------------------------------
# Response side: parse tool calls out of model text
# ---------------------------------------------------------------------------

def _coerce_call(obj: Any) -> dict[str, Any] | None:
    """Normalize a parsed object into {name, arguments:dict} or None."""
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("function")
    if not name or not isinstance(name, str):
        return None
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = {"_raw": args}
    if not isinstance(args, dict):
        args = {"_value": args}
    return {"name": name, "arguments": args}


def _json_objects(text: str) -> list[Any]:
    """Yield every top-level JSON object found in `text` via a brace scan +
    raw_decode (tolerant of surrounding prose)."""
    out: list[Any] = []
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(text, i)
            out.append(obj)
            i = end
        except json.JSONDecodeError:
            i += 1
    return out


def _parse_python_call(expr: str) -> dict[str, Any] | None:
    """Parse a python-style call `fn(a=1, b="x")` into {name, arguments}."""
    try:
        node = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None
    call = node.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    args: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            args[kw.arg] = None
    for idx, a in enumerate(call.args):
        try:
            args[f"arg{idx}"] = ast.literal_eval(a)
        except (ValueError, SyntaxError):
            args[f"arg{idx}"] = None
    return {"name": call.func.id, "arguments": args}


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract tool calls from model output.

    Returns (clean_text, calls) where `calls` is a list of {name, arguments:dict}
    and `clean_text` is the input with the recognized tool-call spans removed.
    Layered: primary `<tool_call>` tags, then `<|python_*|>` tokens, then fenced
    or bare JSON objects carrying a name. Malformed fragments are skipped.
    """
    if not text:
        return text or "", []
    calls: list[dict[str, Any]] = []

    # 1) Primary: <tool_call> ... </tool_call> (possibly several, parallel calls).
    matched_spans: list[tuple[int, int]] = []
    for m in _TOOL_CALL_RE.finditer(text):
        for obj in _json_objects(m.group(1)) or [_try_json(m.group(1))]:
            call = _coerce_call(obj)
            if call:
                calls.append(call)
                matched_spans.append(m.span())
                break
    if calls:
        return _strip_spans(text, matched_spans), calls

    # 1b) Unclosed <tool_call> (truncated by max_tokens).
    m = _TOOL_CALL_OPEN_RE.search(text)
    if m:
        for obj in _json_objects(m.group(1)):
            call = _coerce_call(obj)
            if call:
                calls.append(call)
                return _strip_spans(text, [(m.start(), len(text))]), calls

    # 2) Defensive: nanochat <|python_start|> ... <|python_end|>.
    spans: list[tuple[int, int]] = []
    for m in _PYTHON_TOKEN_RE.finditer(text):
        inner = m.group(1)
        call = None
        for obj in _json_objects(inner):
            call = _coerce_call(obj)
            if call:
                break
        if call is None:
            call = _parse_python_call(inner)
        if call:
            calls.append(call)
            spans.append(m.span())
    if calls:
        return _strip_spans(text, spans), calls

    # 3) Last resort: fenced or bare JSON object with a name + arguments.
    for m in _FENCE_RE.finditer(text):
        for obj in _json_objects(m.group(1)):
            call = _coerce_call(obj)
            if call and "arguments" in (obj if isinstance(obj, dict) else {}):
                calls.append(call)
                spans.append(m.span())
                break
    if calls:
        return _strip_spans(text, spans), calls

    for obj in _json_objects(text):
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            call = _coerce_call(obj)
            if call:
                calls.append(call)
    # Bare-JSON path: don't try to surgically strip; return original text.
    return text, calls


def _try_json(s: str) -> Any:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text.strip()
    out = []
    last = 0
    for start, end in sorted(spans):
        out.append(text[last:start])
        last = end
    out.append(text[last:])
    return "".join(out).strip()


def to_openai_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape parsed calls into the OpenAI `tool_calls` array. Ids are index-based
    so output is deterministic (important for tests)."""
    return [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": c["name"],
                "arguments": json.dumps(c.get("arguments") or {}, ensure_ascii=False),
            },
        }
        for i, c in enumerate(calls)
    ]
