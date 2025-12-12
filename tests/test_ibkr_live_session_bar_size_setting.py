"""Unit tests for src.ibkr_live_session frequency mapping.

These tests are pure-unit (no live IBKR connectivity).
"""

from __future__ import annotations

import pytest

from src.ibkr_live_session import _frequency_to_bar_size_setting


@pytest.mark.parametrize(
    "freq, expected",
    [
        ("1min", "1 min"),
        ("1m", "1 min"),
        ("1 min", "1 min"),
        ("5min", "5 mins"),
        ("5m", "5 mins"),
        ("5 min", "5 mins"),
        ("15min", "15 mins"),
        ("15m", "15 mins"),
        ("30min", "30 mins"),
        ("60min", "1 hour"),
        ("1h", "1 hour"),
        ("240min", "4 hours"),
        ("4h", "4 hours"),
    ],
)
def test_frequency_to_bar_size_setting(freq: str, expected: str) -> None:
    assert _frequency_to_bar_size_setting(freq) == expected


def test_frequency_to_bar_size_setting_raises_for_unknown() -> None:
    with pytest.raises(ValueError):
        _frequency_to_bar_size_setting("2min")
