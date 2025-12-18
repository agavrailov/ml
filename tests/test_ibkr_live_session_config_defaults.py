from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_repo_root(tmp_path: Path):
    """Provide a temporary repo root with configs/active.json."""
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    with patch("src.core.config_library._get_repo_root", return_value=tmp_path):
        yield tmp_path


def test_resolve_uses_active_meta_when_cli_missing(mock_repo_root: Path) -> None:
    from src.ibkr_live_session import _resolve_symbol_frequency_from_active_config

    active = {
        "meta": {
            "symbol": "NVDA",
            "frequency": "15min",
        },
        "strategy": {},
    }

    symbol, freq = _resolve_symbol_frequency_from_active_config(
        cli_symbol=None,
        cli_frequency=None,
        active_config=active,
    )

    assert symbol == "NVDA"
    assert freq == "15min"


def test_resolve_cli_overrides_active_meta() -> None:
    from src.ibkr_live_session import _resolve_symbol_frequency_from_active_config

    active = {"meta": {"symbol": "NVDA", "frequency": "60min"}}

    symbol, freq = _resolve_symbol_frequency_from_active_config(
        cli_symbol="AAPL",
        cli_frequency="5min",
        active_config=active,
    )

    assert symbol == "AAPL"
    assert freq == "5min"


def test_resolve_errors_when_missing_everywhere() -> None:
    from src.ibkr_live_session import _resolve_symbol_frequency_from_active_config

    with pytest.raises(SystemExit):
        _resolve_symbol_frequency_from_active_config(
            cli_symbol=None,
            cli_frequency=None,
            active_config={},
        )


def test_config_library_read_active_config_with_meta(mock_repo_root: Path) -> None:
    from src.core import config_library

    active_path = mock_repo_root / "configs" / "active.json"
    active_path.write_text(
        json.dumps({"meta": {"symbol": "NVDA", "frequency": "60min"}, "strategy": {}}, indent=2),
        encoding="utf-8",
    )

    data = config_library.read_active_config()
    assert isinstance(data, dict)
    assert data["meta"]["frequency"] == "60min"


def test_no_update_warning_threshold_seconds_is_frequency_aware() -> None:
    from src.ibkr_live_session import _no_update_warning_threshold_seconds

    # For 60min, threshold = max(300, 2 * 3600) = 7200s.
    assert _no_update_warning_threshold_seconds("60min") == 7200.0

    # For 240min, threshold = 2 * 240 * 60 = 28800s.
    assert _no_update_warning_threshold_seconds("240min") == 28800.0

    # Unknown values fall back to the base 5 minutes.
    assert _no_update_warning_threshold_seconds("weird") == 300.0
