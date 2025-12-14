"""Tests for src.ui.app entrypoint.

After Phase 3 completion, src.ui.app directly orchestrates tabs without delegation.
This test validates that the module imports successfully and has expected structure.
"""

from __future__ import annotations

import src.ui.app as ui_app


def test_ui_app_imports_successfully():
    """src.ui.app should import without errors."""
    assert ui_app is not None


def test_ui_app_has_expected_helpers():
    """src.ui.app should have helper functions for parameter management."""
    assert hasattr(ui_app, "_build_default_params_df")
    assert hasattr(ui_app, "_load_params_grid")
    assert hasattr(ui_app, "_save_params_grid")
    assert hasattr(ui_app, "_save_strategy_defaults_to_config")
    assert hasattr(ui_app, "_load_strategy_defaults")
    assert hasattr(ui_app, "_run_backtest")
