"""Smoke eval for function calling against the REAL model.

This is the live counterpart to test_tool_calling.py (which is model-free). It
proves the bug is fixed end-to-end: with the Hermes formatter + few-shot, the
model emits parseable <tool_call> JSON instead of looping on the raw instruction.

What it guarantees when run:
  * NO case loops (the original bug was the Nepali system prompt echoed forever).
  * Any tool call the model emits is well-formed and names a real tool.
  * The specific Nepali regression produces a tool call (no loop).
  * Across tool-requiring cases, the tool-call production rate clears a floor.

Strict per-argument accuracy lives in the BFCL-style harness (tests/bfcl/), not
here — tool *selection* quality is a property of the 0.5B model.

Skipped unless HIMALAYA_EVAL_LLAMA_URL points at a running llama-server. Run:

  # start a server (prebuilt fork binary works):
  llama-server -m himalayagpt-0.5b-it-bf16.gguf --port 9990 \\
    --temp 0.2 --top-k 40 --repeat-penalty 1.08
  HIMALAYA_EVAL_LLAMA_URL=http://127.0.0.1:9990 \\
    uv run --directory api/master pytest ../tests/test_tool_calling_eval.py -v -s
"""
from __future__ import annotations

import pytest

from . import eval_support as ev
from .loop_detection import looping

pytestmark = pytest.mark.skipif(
    not ev.have_server(), reason="set HIMALAYA_EVAL_LLAMA_URL to a running llama-server to run")

CASES = ev.load_cases()
TOOL_CASES = [c for c in CASES if not c["expect"].get("no_tool")]
# Single-shot floor for tool-requiring cases. The 0.5B model is stochastic and
# sometimes answers in prose instead of calling (a model-quality limit, not a
# format bug) — this floor exists to catch a FORMATTER regression (rate → ~0),
# not to assert the model is good. The printed rate is the honest number.
PRODUCTION_FLOOR = 0.3


@pytest.fixture(scope="module")
def results() -> dict[str, tuple[str, list[dict]]]:
    """Run every fixture once against the model; share across tests."""
    out: dict[str, tuple[str, list[dict]]] = {}
    for case in CASES:
        text, calls = ev.run_tool_case(case["tools"], [{"role": "user", "content": case["query"]}])
        out[case["id"]] = (text, calls)
    return out


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_no_loop_and_wellformed_calls(case, results):
    """Core regression: output must not loop, and any emitted tool call must be
    well-formed and reference one of the provided tools."""
    text, calls = results[case["id"]]
    is_loop, sample = looping(text)
    assert not is_loop, f"[{case['id']}] response is LOOPING: {sample!r}\n---\n{text}"
    names = ev.tool_names(case)
    for c in calls:
        assert c["name"] in names, f"[{case['id']}] called unknown tool {c['name']!r} (have {names})"
        assert isinstance(c["arguments"], dict), f"[{case['id']}] arguments not a dict: {c['arguments']!r}"


def test_nepali_regression():
    """The exact reported bug was the Nepali tool system-prompt echoed forever in
    a `---`-separated loop. The HARD guarantee here is that the model NEVER
    loops/garbles on the Nepali tool prompts. We also demonstrate the Nepali tool
    SET is reachable — across retries at least one Nepali case emits a valid tool
    call. (Whether a given query triggers a call vs a plain answer is tool-recall,
    a 0.5B model-quality matter measured by test_tool_call_production_rate.)"""
    nepal_cases = [c for c in CASES if c["category"] == "regression"]
    assert nepal_cases, "no regression fixtures found"
    any_valid_call = False
    for case in nepal_cases:
        names = ev.tool_names(case)
        for attempt in range(3):
            text, calls = ev.run_tool_case(case["tools"], [{"role": "user", "content": case["query"]}])
            is_loop, sample = looping(text)
            assert not is_loop, (
                f"[{case['id']}] Nepali bug REGRESSED (looping) on attempt {attempt}: "
                f"{sample!r}\n---\n{text}")
            if calls and calls[0]["name"] in names:
                any_valid_call = True
                break
    assert any_valid_call, "no Nepali-tool case produced a valid tool call across retries"


def test_tool_call_production_rate(results):
    """Across tool-requiring cases, a valid tool call must be produced at least
    PRODUCTION_FLOOR of the time. Prints a per-case report for visibility."""
    produced, report = 0, []
    for case in TOOL_CASES:
        text, calls = results[case["id"]]
        names = ev.tool_names(case)
        ok = bool(calls) and calls[0]["name"] in names
        produced += ok
        got = [c["name"] for c in calls] or "no-call"
        report.append(f"  {'OK ' if ok else '   '} {case['id']:24s} -> {got}")
    rate = produced / max(1, len(TOOL_CASES))
    print(f"\ntool-call production: {produced}/{len(TOOL_CASES)} = {rate:.0%}")
    print("\n".join(report))
    assert rate >= PRODUCTION_FLOOR, (
        f"production rate {rate:.0%} below floor {PRODUCTION_FLOOR:.0%} — "
        "formatter (few-shot) or model regression")
