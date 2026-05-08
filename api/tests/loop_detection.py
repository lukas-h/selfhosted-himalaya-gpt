"""Detect repetition / mode-collapse in a generated string.

Used by test_no_loop_e2e.py to assert that a deployed master endpoint
doesn't return looping output for a default-sampling request.

The detector uses a sliding character n-gram. If any n-character window
appears more than `max_repeats` times in the generation, it's a loop.
That catches both whole-sentence loops ("X. X. X.") and the harder
phrase-level loops ("very strong sense ... very strong sense ..."). It
does *not* flag legitimate repetition like list bullets or normal
sentence starters.
"""
from __future__ import annotations

from collections import Counter


def looping(text: str, *, ngram: int = 60, max_repeats: int = 3) -> tuple[bool, str | None]:
    """Return (is_looping, sample_substring).

    Defaults: any 60-char window that appears more than 3 times → loop.
    60 chars is a typical sentence fragment ("strong, independent
    government. He was a strong believer in"); any text with a fragment
    that long appearing 4+ times is degenerate.
    """
    if len(text) < ngram * 4:
        return False, None
    windows = [text[i : i + ngram] for i in range(0, len(text) - ngram + 1)]
    counts = Counter(windows)
    most, freq = counts.most_common(1)[0]
    if freq > max_repeats:
        return True, most
    return False, None


def loop_metrics(text: str, ngram: int = 60) -> dict:
    """Diagnostic numbers without the binary flag — for logging."""
    if len(text) < ngram:
        return {"chars": len(text), "max_window_repeats": 0, "unique_windows": 0}
    windows = [text[i : i + ngram] for i in range(0, len(text) - ngram + 1)]
    counts = Counter(windows)
    return {
        "chars": len(text),
        "max_window_repeats": counts.most_common(1)[0][1] if counts else 0,
        "unique_windows": len(counts),
        "total_windows": len(windows),
    }
