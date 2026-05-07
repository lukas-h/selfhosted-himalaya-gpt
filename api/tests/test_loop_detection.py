"""Unit tests for the loop detector itself — fast, no network."""
from __future__ import annotations

from .loop_detection import loop_metrics, looping


def test_clean_text_not_loop() -> None:
    text = (
        "Neil Armstrong was the first person to walk on the Moon, on "
        "20 July 1969 during NASA's Apollo 11 mission. He was a US Navy "
        "aviator before joining the astronaut corps. Buzz Aldrin walked "
        "on the Moon shortly after him; Michael Collins stayed in lunar orbit."
    )
    is_loop, _ = looping(text)
    assert not is_loop


def test_obvious_loop() -> None:
    fragment = (
        "He was a strong believer in the importance of individual freedom"
        " and the need for a strong, independent government. "
    )
    text = fragment * 8
    is_loop, sample = looping(text)
    assert is_loop
    assert sample is not None
    assert "strong believer" in sample


def test_short_text_never_loops() -> None:
    # Below the minimum length, we shouldn't false-positive on something tiny.
    is_loop, _ = looping("Yes.")
    assert not is_loop


def test_phrase_loop_caught_even_with_some_variation() -> None:
    # The exact regression we hit in production: same long fragment repeats
    # several times, with a tiny prefix variation between repeats.
    fragment = (
        "He was a strong believer in the importance of individual freedom"
        " and the need for a strong, independent government."
    )
    text = "\n\n".join(fragment for _ in range(6))
    is_loop, _ = looping(text)
    assert is_loop


def test_metrics_smoke() -> None:
    m = loop_metrics("a" * 200)
    assert m["chars"] == 200
    assert m["unique_windows"] == 1  # all windows identical
