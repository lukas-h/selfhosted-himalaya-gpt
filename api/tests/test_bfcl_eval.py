"""BFCL-style accuracy test (env-gated).

Scores the vendored fixtures by function-name + argument structural match, per
category, and asserts an overall floor. The printed report is the honest number;
the floor only guards against a formatter/parser regression (accuracy collapsing
toward zero), not the model's absolute quality.

Run:
  HIMALAYA_EVAL_LLAMA_URL=http://127.0.0.1:9990 \\
    uv run --directory api/master pytest ../tests/test_bfcl_eval.py -v -s

Point at a larger external set (e.g. a converted slice of
interstellarninja/hermes-function-calling-v1) with HIMALAYA_BFCL_CASES=path.json,
or run the harness directly:  python api/tests/bfcl/harness.py --base-url ...
"""
from __future__ import annotations

import os

import pytest

from . import eval_support as ev
from .bfcl import harness

pytestmark = pytest.mark.skipif(
    not ev.have_server(), reason="set HIMALAYA_EVAL_LLAMA_URL to a running llama-server to run")

# best-of-2 by default to dampen the 0.5B's run-to-run variance for a stable gate.
ATTEMPTS = int(os.environ.get("HIMALAYA_BFCL_ATTEMPTS", "2"))
# Regression guard only (measured ~42% best-of-2 on Q8). The printed per-category
# report is the honest signal; this floor catches a formatter/parser collapse.
OVERALL_FLOOR = 0.3


def _cases():
    return harness.load_cases(os.environ.get("HIMALAYA_BFCL_CASES"))


def test_bfcl_accuracy():
    report = harness.score_cases(_cases(), attempts=ATTEMPTS)
    print("\n" + harness.format_report(report))
    assert report["overall"]["acc"] >= OVERALL_FLOOR, (
        f"overall accuracy {report['overall']['acc']:.0%} below floor {OVERALL_FLOOR:.0%} "
        "— likely a formatter/parser regression")
