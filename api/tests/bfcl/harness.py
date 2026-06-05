"""BFCL-style function-calling accuracy harness.

Inspired by the Berkeley Function-Calling Leaderboard (gorilla.cs.berkeley.edu):
run each case through the model (via the production Hermes formatter + parser),
then score the emitted call by AST/structural match — function name plus an
argument-subset match with light type/format flexibility — and report accuracy
per BFCL-style category (simple / multiple / parallel / irrelevance / ...).

This is "BFCL-style", not a port of BFCL: the categories and scorer follow the
same ideas, over a curated, vendored case set (api/tests/fixtures/tool_calling/)
that includes the reported Nepali regression. Point it at a bigger set — e.g. a
slice of interstellarninja/hermes-function-calling-v1 converted to the same case
schema — with `--cases file.jsonl` or HIMALAYA_BFCL_CASES.

Case schema (one object):
    {
      "id": "...", "category": "simple|multiple|parallel|irrelevance|...",
      "tools": [ <OpenAI tool> ],
      "query": "<user message>",
      "expect": {
        "function": "name"            # exact tool expected, OR
        "function_any_of": ["a","b"], # any of these is acceptable
        "args_contains": {...},       # these args must appear (subset match)
        "min_calls": 2,               # parallel: at least N valid calls
        "no_tool": true               # irrelevance: model must NOT call a tool
      }
    }

CLI:
    python api/tests/bfcl/harness.py --base-url http://127.0.0.1:9990 [--attempts 1]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))

import eval_support as ev  # noqa: E402
from loop_detection import looping  # noqa: E402


def _match_value(expected: Any, got: Any) -> bool:
    """Lenient value equality: case-insensitive substring for strings, numeric
    equality across int/float/str, else strict =="""
    if isinstance(expected, str) and isinstance(got, str):
        e, g = expected.strip().lower(), got.strip().lower()
        return e == g or e in g or g in e
    if isinstance(expected, bool) or isinstance(got, bool):
        return expected == got
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        return float(expected) == float(got)
    if isinstance(expected, (int, float)) and isinstance(got, str):
        try:
            return float(expected) == float(got)
        except ValueError:
            return str(expected) == got
    return expected == got


def _args_match(expected_args: dict, got_args: Any) -> bool:
    """Subset match: every expected key/value must be present in got_args."""
    if not isinstance(got_args, dict):
        return False
    for k, v in (expected_args or {}).items():
        if k not in got_args or not _match_value(v, got_args[k]):
            return False
    return True


def score_case(case: dict, text: str, calls: list[dict]) -> tuple[bool, str]:
    """Return (correct, note) for one model response."""
    exp = case.get("expect", {})
    if looping(text)[0]:
        return False, "LOOPED"
    if exp.get("no_tool"):
        return (len(calls) == 0), ("ok" if not calls else f"unexpected {[c['name'] for c in calls]}")
    if not calls:
        return False, "no call produced"
    names = {c["name"] for c in calls}
    want = None
    if "function" in exp:
        want = {exp["function"]}
    elif "function_any_of" in exp:
        want = set(exp["function_any_of"])
    if want and not (names & want):
        return False, f"wrong tool {sorted(names)} (want {sorted(want)})"
    if exp.get("min_calls") and len(calls) < exp["min_calls"]:
        return False, f"only {len(calls)} call(s) (want >= {exp['min_calls']})"
    if exp.get("args_contains"):
        target = next((c for c in calls if not want or c["name"] in want), calls[0])
        if not _args_match(exp["args_contains"], target.get("arguments")):
            return False, f"args {target.get('arguments')} !⊇ {exp['args_contains']}"
    return True, "ok"


def score_cases(cases: list[dict], *, base_url: str | None = None, attempts: int = 1) -> dict:
    """Run + score every case. attempts>1 = best-of-N (counts correct if any
    attempt is correct) to dampen the 0.5B model's run-to-run variance."""
    cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    details = []
    for case in cases:
        ok, note, got = False, "", []
        for _ in range(max(1, attempts)):
            text, calls = ev.run_tool_case(
                case["tools"], [{"role": "user", "content": case["query"]}], base=base_url)
            ok, note = score_case(case, text, calls)
            got = [c["name"] for c in calls]
            if ok:
                break
        c = case.get("category", "other")
        cat[c][0] += int(ok)
        cat[c][1] += 1
        details.append({"id": case["id"], "category": c, "ok": ok, "note": note, "calls": got})
    per_cat = {k: {"correct": v[0], "n": v[1], "acc": (v[0] / v[1] if v[1] else 0.0)}
               for k, v in sorted(cat.items())}
    ok_total = sum(v[0] for v in cat.values())
    n_total = sum(v[1] for v in cat.values())
    return {
        "per_category": per_cat,
        "overall": {"correct": ok_total, "n": n_total, "acc": (ok_total / n_total if n_total else 0.0)},
        "details": details,
        "attempts": attempts,
    }


def format_report(report: dict) -> str:
    lines = [f"BFCL-style accuracy (best-of-{report['attempts']})", "=" * 46]
    for cat, s in report["per_category"].items():
        lines.append(f"  {cat:14s} {s['correct']:2d}/{s['n']:2d}   {s['acc']:.0%}")
    o = report["overall"]
    lines.append("-" * 46)
    lines.append(f"  {'OVERALL':14s} {o['correct']:2d}/{o['n']:2d}   {o['acc']:.0%}")
    lines.append("")
    for d in report["details"]:
        mark = "OK " if d["ok"] else " x "
        note = "" if d["ok"] else f"  <- {d['note']}"
        lines.append(f"  {mark}[{d['category']:11s}] {d['id']:22s} {d['calls'] or 'no-call'}{note}")
    return "\n".join(lines)


def load_cases(path: str | None) -> list[dict]:
    if not path:
        return ev.load_cases()
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in txt.splitlines() if line.strip()]
    data = json.loads(txt)
    return data if isinstance(data, list) else [data]


def main() -> None:
    ap = argparse.ArgumentParser(description="BFCL-style function-calling accuracy harness")
    ap.add_argument("--base-url", default=ev.LLAMA_URL or "http://127.0.0.1:9990",
                    help="llama-server base URL (default $HIMALAYA_EVAL_LLAMA_URL or :9990)")
    ap.add_argument("--cases", default=None, help="JSON/JSONL case file (default: vendored fixtures)")
    ap.add_argument("--attempts", type=int, default=1, help="best-of-N per case (default 1 = single-shot)")
    args = ap.parse_args()
    report = score_cases(load_cases(args.cases), base_url=args.base_url, attempts=args.attempts)
    print(format_report(report))


if __name__ == "__main__":
    main()
