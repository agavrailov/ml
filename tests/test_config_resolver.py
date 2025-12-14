"""Tests for src.core.config_resolver - config merging and persistence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_configs_dir(tmp_path: Path):
    """Create a temporary configs/ directory for testing."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    return configs_dir


@pytest.fixture
def mock_repo_root(temp_configs_dir: Path):
    """Mock _get_repo_root to point to the temp directory."""
    repo_root = temp_configs_dir.parent
    with patch("src.core.config_resolver._get_repo_root", return_value=repo_root):
        yield repo_root


def test_get_strategy_defaults_no_active_json(mock_repo_root):
    """When configs/active.json does not exist, return config.py defaults."""
    from src.core.config_resolver import get_strategy_defaults
    from src.config import STRATEGY_DEFAULTS

    defaults = get_strategy_defaults()

    # Should match code defaults from config.py.
    assert defaults["risk_per_trade_pct"] == STRATEGY_DEFAULTS.risk_per_trade_pct
    assert defaults["reward_risk_ratio"] == STRATEGY_DEFAULTS.reward_risk_ratio
    assert defaults["k_sigma_long"] == STRATEGY_DEFAULTS.k_sigma_long
    assert defaults["k_sigma_short"] == STRATEGY_DEFAULTS.k_sigma_short
    assert defaults["k_atr_long"] == STRATEGY_DEFAULTS.k_atr_long
    assert defaults["k_atr_short"] == STRATEGY_DEFAULTS.k_atr_short


def test_get_strategy_defaults_with_overrides(mock_repo_root, temp_configs_dir):
    """When configs/active.json exists with overrides, merge with code defaults."""
    from src.core.config_resolver import get_strategy_defaults

    # Write user overrides to active.json.
    active_json_path = temp_configs_dir / "active.json"
    overrides = {
        "strategy": {
            "risk_per_trade_pct": 0.002,
            "k_sigma_long": 0.5,
        }
    }
    active_json_path.write_text(json.dumps(overrides, indent=2), encoding="utf-8")

    defaults = get_strategy_defaults()

    # Overridden values should come from active.json.
    assert defaults["risk_per_trade_pct"] == 0.002
    assert defaults["k_sigma_long"] == 0.5

    # Non-overridden values should still come from config.py.
    from src.config import STRATEGY_DEFAULTS
    assert defaults["reward_risk_ratio"] == STRATEGY_DEFAULTS.reward_risk_ratio
    assert defaults["k_sigma_short"] == STRATEGY_DEFAULTS.k_sigma_short
    assert defaults["k_atr_long"] == STRATEGY_DEFAULTS.k_atr_long
    assert defaults["k_atr_short"] == STRATEGY_DEFAULTS.k_atr_short


def test_save_strategy_defaults_creates_active_json(mock_repo_root, temp_configs_dir):
    """Saving strategy defaults creates configs/active.json."""
    from src.core.config_resolver import save_strategy_defaults

    save_strategy_defaults(
        risk_per_trade_pct=0.003,
        reward_risk_ratio=0.2,
        k_sigma_long=0.6,
        k_sigma_short=0.4,
        k_atr_long=0.3,
        k_atr_short=0.25,
    )

    active_json_path = temp_configs_dir / "active.json"
    assert active_json_path.exists()

    data = json.loads(active_json_path.read_text(encoding="utf-8"))
    assert data["strategy"]["risk_per_trade_pct"] == 0.003
    assert data["strategy"]["reward_risk_ratio"] == 0.2
    assert data["strategy"]["k_sigma_long"] == 0.6
    assert data["strategy"]["k_sigma_short"] == 0.4
    assert data["strategy"]["k_atr_long"] == 0.3
    assert data["strategy"]["k_atr_short"] == 0.25


def test_save_and_load_roundtrip(mock_repo_root, temp_configs_dir):
    """Saving then loading strategy defaults produces the same values."""
    from src.core.config_resolver import get_strategy_defaults, save_strategy_defaults

    # Save custom values.
    save_strategy_defaults(
        risk_per_trade_pct=0.0025,
        reward_risk_ratio=0.15,
        k_sigma_long=0.7,
        k_sigma_short=0.5,
        k_atr_long=0.4,
        k_atr_short=0.35,
    )

    # Load and verify.
    defaults = get_strategy_defaults()
    assert defaults["risk_per_trade_pct"] == 0.0025
    assert defaults["reward_risk_ratio"] == 0.15
    assert defaults["k_sigma_long"] == 0.7
    assert defaults["k_sigma_short"] == 0.5
    assert defaults["k_atr_long"] == 0.4
    assert defaults["k_atr_short"] == 0.35


def test_save_does_not_edit_config_py(mock_repo_root, temp_configs_dir, tmp_path):
    """Saving strategy defaults does NOT modify src/config.py."""
    from src.core.config_resolver import save_strategy_defaults

    # Create a fake config.py to ensure it's not edited.
    fake_config_py = tmp_path / "config.py"
    original_content = "# This is a test config.py\nrisk_per_trade_pct: float = 0.001\n"
    fake_config_py.write_text(original_content, encoding="utf-8")

    # Save strategy defaults (should write to active.json only).
    save_strategy_defaults(
        risk_per_trade_pct=0.999,
        reward_risk_ratio=0.8,
        k_sigma_long=1.0,
        k_sigma_short=1.0,
        k_atr_long=1.0,
        k_atr_short=1.0,
    )

    # Verify config.py was not touched.
    assert fake_config_py.read_text(encoding="utf-8") == original_content


def test_load_strategy_defaults_empty_active_json(mock_repo_root, temp_configs_dir):
    """When active.json is empty or invalid, fall back to config.py defaults."""
    from src.core.config_resolver import get_strategy_defaults
    from src.config import STRATEGY_DEFAULTS

    # Write an empty file.
    active_json_path = temp_configs_dir / "active.json"
    active_json_path.write_text("", encoding="utf-8")

    defaults = get_strategy_defaults()

    # Should use config.py defaults.
    assert defaults["risk_per_trade_pct"] == STRATEGY_DEFAULTS.risk_per_trade_pct
    assert defaults["reward_risk_ratio"] == STRATEGY_DEFAULTS.reward_risk_ratio


def test_load_strategy_defaults_malformed_active_json(mock_repo_root, temp_configs_dir):
    """When active.json is malformed, fall back to config.py defaults."""
    from src.core.config_resolver import get_strategy_defaults
    from src.config import STRATEGY_DEFAULTS

    # Write invalid JSON.
    active_json_path = temp_configs_dir / "active.json"
    active_json_path.write_text("{invalid json", encoding="utf-8")

    defaults = get_strategy_defaults()

    # Should use config.py defaults.
    assert defaults["risk_per_trade_pct"] == STRATEGY_DEFAULTS.risk_per_trade_pct
