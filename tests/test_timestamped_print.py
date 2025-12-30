"""Tests for timestamped print utility."""
from __future__ import annotations

import re
from io import StringIO
from unittest.mock import patch

from src.utils.timestamped_print import ts_print


def test_ts_print_adds_timestamp():
    """Test that ts_print adds timestamp in MM-DD HH:MM:SS format."""
    with patch("sys.stdout", new=StringIO()) as fake_out:
        ts_print("Test message")
        output = fake_out.getvalue()
        
        # Check format: MM-DD HH:MM:SS Test message
        pattern = r"^\d{2}-\d{2} \d{2}:\d{2}:\d{2} Test message\n$"
        assert re.match(pattern, output), f"Output doesn't match expected format: {output!r}"


def test_ts_print_multiple_args():
    """Test that ts_print handles multiple arguments like regular print."""
    with patch("sys.stdout", new=StringIO()) as fake_out:
        ts_print("Value:", 42, "another:", 3.14)
        output = fake_out.getvalue()
        
        # Should contain all arguments
        assert "Value: 42 another: 3.14" in output


def test_ts_print_kwargs():
    """Test that ts_print passes through print kwargs."""
    with patch("sys.stdout", new=StringIO()) as fake_out:
        ts_print("No newline", end="")
        output = fake_out.getvalue()
        
        # Should not end with newline
        assert not output.endswith("\n")
        assert "No newline" in output
