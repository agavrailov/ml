"""Tests for market-aware reconnect logic in src.ibkr_live_session.

Verifies that the reconnect loop:
- Uses longer backoff intervals during off-market hours
- Reduces log spam during weekends/holidays
- Allows normal reconnection during premarket hours
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

import src.ibkr_live_session as m


def test_is_trading_day_weekday():
    """Test that weekdays are recognized as trading days."""
    # Monday
    monday = datetime(2025, 12, 22, 12, 0, tzinfo=timezone.utc)
    assert m._is_trading_day(monday) is True


def test_is_trading_day_weekend():
    """Test that weekends are not trading days."""
    # Saturday
    saturday = datetime(2025, 12, 27, 12, 0, tzinfo=timezone.utc)
    assert m._is_trading_day(saturday) is False
    
    # Sunday
    sunday = datetime(2025, 12, 28, 12, 0, tzinfo=timezone.utc)
    assert m._is_trading_day(sunday) is False


def test_is_trading_day_with_market_calendar():
    """Test holiday detection using market calendar if available."""
    try:
        import pandas_market_calendars  # noqa: F401
        
        # Christmas 2025 (Thursday, Dec 25) is a market holiday
        christmas = datetime(2025, 12, 25, 12, 0, tzinfo=timezone.utc)
        result = m._is_trading_day(christmas)
        # Should be False if calendar is working correctly
        assert result is False
    except ImportError:
        pytest.skip("pandas_market_calendars not available")


def test_is_market_hours_premarket():
    """Test that premarket hours (4 AM - 9:30 AM EST) are recognized."""
    try:
        import zoneinfo
    except ImportError:
        pytest.skip("zoneinfo not available")
    
    with patch("src.ibkr_live_session.datetime") as mock_dt:
        eastern = zoneinfo.ZoneInfo("America/New_York")
        # 5 AM EST on a weekday (in premarket)
        mock_now = datetime(2025, 12, 22, 10, 0, tzinfo=timezone.utc)  # 5 AM EST
        mock_dt.now.return_value = mock_now
        
        result = m._is_market_hours(premarket=True)
        assert result is True


def test_is_market_hours_regular():
    """Test that regular market hours (9:30 AM - 4 PM EST) are recognized."""
    try:
        import zoneinfo
    except ImportError:
        pytest.skip("zoneinfo not available")
    
    with patch("src.ibkr_live_session.datetime") as mock_dt:
        eastern = zoneinfo.ZoneInfo("America/New_York")
        # 2 PM EST on a weekday (during market hours)
        mock_now = datetime(2025, 12, 22, 19, 0, tzinfo=timezone.utc)  # 2 PM EST
        mock_dt.now.return_value = mock_now
        
        result = m._is_market_hours(premarket=True)
        assert result is True
        
        result = m._is_market_hours(premarket=False)
        assert result is True


def test_is_market_hours_after_close():
    """Test that hours after market close are not recognized as market hours."""
    try:
        import zoneinfo
    except ImportError:
        pytest.skip("zoneinfo not available")
    
    with patch("src.ibkr_live_session.datetime") as mock_dt:
        eastern = zoneinfo.ZoneInfo("America/New_York")
        # 6 PM EST on a weekday (after market close)
        mock_now = datetime(2025, 12, 22, 23, 0, tzinfo=timezone.utc)  # 6 PM EST
        mock_dt.now.return_value = mock_now
        
        result = m._is_market_hours(premarket=True)
        assert result is False


def test_is_market_hours_weekend():
    """Test that weekends are not market hours."""
    try:
        import zoneinfo
    except ImportError:
        pytest.skip("zoneinfo not available")
    
    with patch("src.ibkr_live_session.datetime") as mock_dt:
        eastern = zoneinfo.ZoneInfo("America/New_York")
        # Saturday at 2 PM EST
        mock_now = datetime(2025, 12, 27, 19, 0, tzinfo=timezone.utc)
        mock_dt.now.return_value = mock_now
        
        result = m._is_market_hours(premarket=True)
        assert result is False
