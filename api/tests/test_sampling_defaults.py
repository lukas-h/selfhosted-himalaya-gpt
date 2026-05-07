"""Static check: each llama-* service in worker/docker-compose.yml must
pass the recommended sampling defaults to llama-server.

Why: llama-server's built-in defaults disable repeat_penalty (=1.0). The
nanochat 0.5B model loops within ~300 tokens without it. NANOCHAT_GGUF_HANDOVER.md
and the upstream Himalaya AI Colab both recommend
`temperature=0.2 top_k=40 repeat_penalty=1.08`. Per-request fields still
override these defaults.

Run: `pytest api/tests/test_sampling_defaults.py -v`
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPOSE = Path(__file__).resolve().parents[1] / "worker" / "docker-compose.yml"

REQUIRED = {
    "--temp": "0.2",
    "--top-k": "40",
    "--repeat-penalty": "1.08",
}

LLAMA_SERVICES = ["llama-bf16", "llama-q8", "llama-q4"]


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text())


@pytest.mark.parametrize("svc", LLAMA_SERVICES)
def test_llama_service_has_sampling_defaults(compose: dict, svc: str) -> None:
    """Each per-quant llama service must set temp/top-k/repeat-penalty
    on the cmdline so requests without sampling overrides don't loop."""
    cmd = compose["services"][svc]["command"]
    # docker-compose `command:` is a list of alternating flag/value strings
    # — pair them up so we can look up flag values regardless of position.
    pairs = dict(zip(cmd[::2], cmd[1::2], strict=False))
    for flag, expected in REQUIRED.items():
        assert flag in pairs, (
            f"{svc} is missing `{flag}` — without it the model loops on "
            f"long generations. Expected `{flag} {expected}` in command."
        )
        assert pairs[flag] == expected, (
            f"{svc} has `{flag} {pairs[flag]}` but expected `{flag} {expected}`. "
            f"See NANOCHAT_GGUF_HANDOVER.md for the rationale."
        )
