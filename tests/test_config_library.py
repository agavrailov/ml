from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_repo_root(tmp_path: Path):
    """Mock repo root for both config_library and config_resolver."""
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)

    with (
        patch("src.core.config_library._get_repo_root", return_value=tmp_path),
        patch("src.core.config_resolver._get_repo_root", return_value=tmp_path),
    ):
        yield tmp_path


def test_save_candidate_writes_payload_and_manifest(mock_repo_root: Path) -> None:
    from src.core import config_library

    row = config_library.save_candidate(
        symbol="NVDA",
        frequency="60min",
        strategy={
            "risk_per_trade_pct": 0.01,
            "reward_risk_ratio": 2.0,
            "k_sigma_long": 0.7,
            "k_sigma_short": 0.9,
            "k_atr_long": 0.7,
            "k_atr_short": 1.0,
        },
        label="test",
        source="pytest",
        metrics={"sharpe_ratio": 1.23, "n_trades": 10},
    )

    assert row["symbol"] == "NVDA"
    assert row["frequency"] == "60min"
    assert row["id"].startswith("cfg_")

    # Candidate file exists.
    cand_path = mock_repo_root / Path(row["path"])
    assert cand_path.exists()

    payload = json.loads(cand_path.read_text(encoding="utf-8"))
    assert payload["meta"]["id"] == row["id"]
    assert payload["meta"]["symbol"] == "NVDA"
    assert payload["meta"]["frequency"] == "60min"
    assert payload["strategy"]["risk_per_trade_pct"] == 0.01

    # Manifest exists and includes the row.
    manifest_path = mock_repo_root / "configs" / "library" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert any(r.get("id") == row["id"] for r in manifest)


def test_list_candidates_filters(mock_repo_root: Path) -> None:
    from src.core import config_library

    r1 = config_library.save_candidate(
        symbol="NVDA",
        frequency="60min",
        strategy={"risk_per_trade_pct": 0.01, "reward_risk_ratio": 2.0, "k_sigma_long": 0.1, "k_sigma_short": 0.1, "k_atr_long": 0.1, "k_atr_short": 0.1},
    )
    _ = config_library.save_candidate(
        symbol="NVDA",
        frequency="15min",
        strategy={"risk_per_trade_pct": 0.02, "reward_risk_ratio": 1.0, "k_sigma_long": 0.2, "k_sigma_short": 0.2, "k_atr_long": 0.2, "k_atr_short": 0.2},
    )

    rows_60 = config_library.list_candidates(symbol="NVDA", frequency="60min")
    assert len(rows_60) == 1
    assert rows_60[0]["id"] == r1["id"]

    rows_all = config_library.list_candidates(symbol="NVDA")
    assert len(rows_all) == 2


def test_promote_candidate_writes_active_json_and_resolver_reads(mock_repo_root: Path) -> None:
    from src.core import config_library
    from src.core.config_resolver import get_strategy_defaults

    row = config_library.save_candidate(
        symbol="NVDA",
        frequency="60min",
        strategy={
            "risk_per_trade_pct": 0.003,
            "reward_risk_ratio": 0.2,
            "k_sigma_long": 0.6,
            "k_sigma_short": 0.4,
            "k_atr_long": 0.3,
            "k_atr_short": 0.25,
        },
        source="pytest",
    )

    config_library.promote_candidate(row["id"])

    active_path = mock_repo_root / "configs" / "active.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))

    assert active["meta"]["symbol"] == "NVDA"
    assert active["meta"]["frequency"] == "60min"
    assert "promoted_at_utc" in active["meta"]

    # Resolver should ignore meta and read strategy values.
    defaults = get_strategy_defaults()
    assert defaults["risk_per_trade_pct"] == 0.003
    assert defaults["reward_risk_ratio"] == 0.2


def test_list_candidates_falls_back_to_scan_when_manifest_corrupt(mock_repo_root: Path) -> None:
    from src.core import config_library

    row = config_library.save_candidate(
        symbol="NVDA",
        frequency="60min",
        strategy={"risk_per_trade_pct": 0.01, "reward_risk_ratio": 2.0, "k_sigma_long": 0.1, "k_sigma_short": 0.1, "k_atr_long": 0.1, "k_atr_short": 0.1},
    )

    manifest_path = mock_repo_root / "configs" / "library" / "manifest.json"
    manifest_path.write_text("{not json", encoding="utf-8")

    rows = config_library.list_candidates(symbol="NVDA", frequency="60min")
    assert len(rows) == 1
    assert rows[0]["id"] == row["id"]
