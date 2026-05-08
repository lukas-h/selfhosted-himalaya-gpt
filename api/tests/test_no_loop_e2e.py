"""End-to-end regression test: a fresh request to each slug with NO
sampling overrides must not return looping output.

This is the test that would have caught the Neil-Armstrong-loop incident.
It exercises the full path (master → agent → llama-server) so it also
covers any stripping / re-injection of sampling fields between layers.

Skipped unless `HIMALAYA_E2E_BASE_URL` is set — pointing at a deployed
master, e.g. `https://himalayagpt.api.scalabs.ai`.

Run:
  HIMALAYA_E2E_BASE_URL=https://himalayagpt.api.scalabs.ai \\
  HIMALAYA_E2E_TOKEN=momo \\
  pytest api/tests/test_no_loop_e2e.py -v -s
"""
from __future__ import annotations

import json
import os
import urllib.request

import pytest

from .loop_detection import loop_metrics, looping

BASE_URL = os.environ.get("HIMALAYA_E2E_BASE_URL", "")
TOKEN = os.environ.get("HIMALAYA_E2E_TOKEN", "")
SLUGS = os.environ.get(
    "HIMALAYA_E2E_SLUGS", "himalaya-bf16,himalaya-q8,himalaya-q4"
).split(",")

pytestmark = pytest.mark.skipif(
    not BASE_URL or not TOKEN,
    reason="set HIMALAYA_E2E_BASE_URL and HIMALAYA_E2E_TOKEN to run",
)


def _completion(slug: str, prompt: str, *, max_tokens: int = 1500) -> dict:
    """POST a chat completion with NO sampling overrides — exactly the
    shape that triggered the original loop. Returns the parsed body."""
    body = json.dumps({
        "model": slug,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


@pytest.mark.parametrize("slug", SLUGS)
def test_no_loop_default_sampling(slug: str) -> None:
    """The Neil Armstrong regression. With max_tokens=1500 and no sampling
    overrides, the response must terminate cleanly and not contain a
    repeated 60-char fragment."""
    data = _completion(slug, "who was neil armstrong?", max_tokens=1500)
    text = data["choices"][0]["message"]["content"]
    finish = data["choices"][0]["finish_reason"]
    metrics = loop_metrics(text)

    is_loop, sample = looping(text)
    print(f"\n  [{slug}] finish={finish} {metrics}")
    if is_loop:
        print(f"  loop sample: {sample!r}")
        print(f"  full text:\n{text}")

    assert not is_loop, (
        f"[{slug}] response is looping — server-side sampling defaults "
        f"likely missing (--temp / --top-k / --repeat-penalty). "
        f"Repeated fragment: {sample!r}"
    )
    # Soft signal: clean termination is the goal, but a 0.5B model can also
    # legitimately hit max_tokens for a long bio. We only fail on actual
    # looping above; this is just diagnostic.
    if finish == "length" and metrics["max_window_repeats"] >= 2:
        print(
            f"  warning: [{slug}] hit max_tokens with some repetition "
            f"(max_window_repeats={metrics['max_window_repeats']})"
        )
