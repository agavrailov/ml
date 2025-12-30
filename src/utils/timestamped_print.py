"""Utility for timestamped console output.

Provides a simple wrapper around print() that adds timestamps in MM-DD HH:MM:SS format.
"""
from __future__ import annotations

from datetime import datetime


def ts_print(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
    """Print with timestamp prefix in MM-DD HH:MM:SS format.
    
    Usage:
        ts_print("Hello world")  # Output: 12-30 13:45:23 Hello world
        ts_print("Value:", 42)    # Output: 12-30 13:45:23 Value: 42
    """
    timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
    print(timestamp, *args, **kwargs)
