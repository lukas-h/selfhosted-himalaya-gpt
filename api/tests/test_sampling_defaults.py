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
SYCL_ONLY_ENV = {"ONEAPI_DEVICE_SELECTOR", "SYCL_CACHE_PERSISTENT"}
SYCL_ONLY_FLAGS = {"--batch-size", "--ubatch-size"}


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


@pytest.mark.parametrize("svc", LLAMA_SERVICES)
def test_llama_service_uses_vulkan_backend(compose: dict, svc: str) -> None:
    """All deployed quants use Vulkan; SYCL is faster for some q8/q4 cases
    but has been unstable on this nanochat graph."""
    service = compose["services"][svc]
    assert service["build"]["dockerfile"] == ".devops/vulkan.Dockerfile"
    assert service["image"] == "himalaya-llama:vulkan"

    env = service.get("environment") or {}
    assert not (SYCL_ONLY_ENV & set(env)), f"{svc} still has SYCL-only env vars"

    cmd = set(service["command"])
    assert not (SYCL_ONLY_FLAGS & cmd), f"{svc} still has SYCL-only microbatch flags"
