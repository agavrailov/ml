"""Contract tests for job artifacts.

These tests validate that job handlers produce artifacts with stable, expected schemas.
This prevents regressions in artifact structure that would break UI or downstream consumers.

For each job type (train, backtest, optimize, walkforward), we:
1. Run the job handler with minimal valid inputs
2. Validate that required artifact files exist
3. Validate artifact schemas (columns, data types, key fields)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.core.contracts import BacktestRequest, OptimizeRequest, WalkForwardRequest
from src.jobs import store
from src.jobs.handlers import backtest_job, optimize_job, train_job, walkforward_job


@pytest.fixture
def job_id(tmp_path: Path, monkeypatch) -> str:
    """Create a unique job ID for testing."""
    # Override runs_root to point to temp directory
    monkeypatch.setattr(store, "runs_root", lambda: tmp_path / "runs")
    return "test_job_123"


@pytest.fixture
def sample_predictions_csv(tmp_path: Path) -> Path:
    """Create a minimal predictions CSV for testing."""
    csv_path = tmp_path / "predictions.csv"
    # Create minimal valid predictions CSV
    df = pd.DataFrame(
        {
            "Time": pd.date_range("2024-01-01", periods=100, freq="h"),
            "prediction_1step": [100.0 + i * 0.1 for i in range(100)],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


class TestBacktestJobContract:
    """Test backtest job artifact contracts."""

    def test_backtest_produces_required_artifacts(
        self, job_id: str, sample_predictions_csv: Path
    ):
        """Backtest job should produce equity.csv, trades.csv, and metrics.json."""
        request = BacktestRequest(
            frequency="60min",
            prediction_mode="csv",
            start_date="2024-01-01",
            end_date="2024-01-05",
            predictions_csv=str(sample_predictions_csv),
            risk_per_trade_pct=1.0,
            reward_risk_ratio=2.0,
            k_sigma_long=1.0,
            k_sigma_short=1.0,
            k_atr_long=2.0,
            k_atr_short=2.0,
            enable_longs=True,
            allow_shorts=False,
        )

        result = backtest_job.run(job_id, request)

        # Validate artifacts exist
        job_dir = store.get_job_dir(job_id)
        assert (job_dir / "artifacts" / "equity.csv").exists()
        assert (job_dir / "artifacts" / "trades.csv").exists()
        assert (job_dir / "artifacts" / "metrics.json").exists()
        assert (job_dir / "result.json").exists()

    def test_backtest_equity_csv_schema(self, job_id: str, sample_predictions_csv: Path):
        """Equity CSV should have Time and Equity columns."""
        request = BacktestRequest(
            frequency="60min",
            prediction_mode="csv",
            start_date="2024-01-01",
            end_date="2024-01-05",
            predictions_csv=str(sample_predictions_csv),
            risk_per_trade_pct=1.0,
            reward_risk_ratio=2.0,
            k_sigma_long=1.0,
            k_sigma_short=1.0,
            k_atr_long=2.0,
            k_atr_short=2.0,
            enable_longs=True,
            allow_shorts=False,
        )

        backtest_job.run(job_id, request)

        equity_df = pd.read_csv(store.get_job_dir(job_id) / "artifacts" / "equity.csv")
        assert "Time" in equity_df.columns
        assert "Equity" in equity_df.columns
        assert len(equity_df) > 0

    def test_backtest_trades_csv_schema(self, job_id: str, sample_predictions_csv: Path):
        """Trades CSV should have required trade fields."""
        request = BacktestRequest(
            frequency="60min",
            prediction_mode="csv",
            start_date="2024-01-01",
            end_date="2024-01-05",
            predictions_csv=str(sample_predictions_csv),
            risk_per_trade_pct=1.0,
            reward_risk_ratio=2.0,
            k_sigma_long=1.0,
            k_sigma_short=1.0,
            k_atr_long=2.0,
            k_atr_short=2.0,
            enable_longs=True,
            allow_shorts=False,
        )

        backtest_job.run(job_id, request)

        trades_df = pd.read_csv(store.get_job_dir(job_id) / "artifacts" / "trades.csv")
        expected_cols = {"EntryTime", "ExitTime", "Direction", "Size", "PnL"}
        assert expected_cols.issubset(set(trades_df.columns))

    def test_backtest_metrics_json_schema(
        self, job_id: str, sample_predictions_csv: Path
    ):
        """Metrics JSON should have required performance metrics."""
        request = BacktestRequest(
            frequency="60min",
            prediction_mode="csv",
            start_date="2024-01-01",
            end_date="2024-01-05",
            predictions_csv=str(sample_predictions_csv),
            risk_per_trade_pct=1.0,
            reward_risk_ratio=2.0,
            k_sigma_long=1.0,
            k_sigma_short=1.0,
            k_atr_long=2.0,
            k_atr_short=2.0,
            enable_longs=True,
            allow_shorts=False,
        )

        backtest_job.run(job_id, request)

        metrics_path = store.get_job_dir(job_id) / "artifacts" / "metrics.json"
        with metrics_path.open() as f:
            metrics = json.load(f)

        expected_metrics = {
            "total_return",
            "cagr",
            "max_drawdown",
            "sharpe_ratio",
            "win_rate",
            "n_trades",
            "final_equity",
        }
        assert expected_metrics.issubset(set(metrics.keys()))
        assert isinstance(metrics["total_return"], (int, float))
        assert isinstance(metrics["sharpe_ratio"], (int, float))
        assert isinstance(metrics["n_trades"], int)


class TestOptimizeJobContract:
    """Test optimize job artifact contracts."""

    def test_optimize_produces_required_artifacts(
        self, job_id: str, sample_predictions_csv: Path
    ):
        """Optimize job should produce results.csv and result.json with summary."""
        request = OptimizeRequest(
            frequency="60min",
            start_date="2024-01-01",
            end_date="2024-01-05",
            predictions_csv=str(sample_predictions_csv),
            trade_side="Long only",
            param_grid={
                "k_sigma_long": {"start": 0.5, "stop": 1.5, "step": 0.5},
                "k_sigma_short": 1.0,
                "k_atr_long": {"start": 1.0, "stop": 2.0, "step": 1.0},
                "k_atr_short": 2.0,
                "risk_per_trade_pct": 1.0,
                "reward_risk_ratio": 2.0,
            },
        )

        optimize_job.run(job_id, request)

        job_dir = store.get_job_dir(job_id)
        assert (job_dir / "artifacts" / "results.csv").exists()
        assert (job_dir / "result.json").exists()

    def test_optimize_results_csv_schema(
        self, job_id: str, sample_predictions_csv: Path
    ):
        """Results CSV should have parameter columns and metric columns."""
        request = OptimizeRequest(
            frequency="60min",
            start_date="2024-01-01",
            end_date="2024-01-05",
            predictions_csv=str(sample_predictions_csv),
            trade_side="Long only",
            param_grid={
                "k_sigma_long": {"start": 0.5, "stop": 1.5, "step": 0.5},
                "k_sigma_short": 1.0,
                "k_atr_long": 2.0,
                "k_atr_short": 2.0,
                "risk_per_trade_pct": 1.0,
                "reward_risk_ratio": 2.0,
            },
        )

        optimize_job.run(job_id, request)

        results_df = pd.read_csv(
            store.get_job_dir(job_id) / "artifacts" / "results.csv"
        )

        # Should have parameter columns
        param_cols = {
            "k_sigma_long",
            "k_sigma_short",
            "k_atr_long",
            "k_atr_short",
            "risk_per_trade_pct",
            "reward_risk_ratio",
        }
        assert param_cols.issubset(set(results_df.columns))

        # Should have metric columns
        metric_cols = {
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "n_trades",
        }
        assert metric_cols.issubset(set(results_df.columns))

        # Should be sorted by sharpe_ratio descending
        assert len(results_df) > 1
        assert results_df["sharpe_ratio"].iloc[0] >= results_df["sharpe_ratio"].iloc[1]

    def test_optimize_result_json_schema(
        self, job_id: str, sample_predictions_csv: Path
    ):
        """Result.json should have summary with best_sharpe and n_runs."""
        request = OptimizeRequest(
            frequency="60min",
            start_date="2024-01-01",
            end_date="2024-01-05",
            predictions_csv=str(sample_predictions_csv),
            trade_side="Long only",
            param_grid={
                "k_sigma_long": {"start": 0.5, "stop": 1.0, "step": 0.5},
                "k_sigma_short": 1.0,
                "k_atr_long": 2.0,
                "k_atr_short": 2.0,
                "risk_per_trade_pct": 1.0,
                "reward_risk_ratio": 2.0,
            },
        )

        optimize_job.run(job_id, request)

        result_path = store.get_job_dir(job_id) / "result.json"
        with result_path.open() as f:
            result = json.load(f)

        assert "summary" in result
        assert "best_sharpe" in result["summary"]
        assert "best_total_return" in result["summary"]
        assert "n_runs" in result["summary"]
        assert isinstance(result["summary"]["n_runs"], int)
        assert result["summary"]["n_runs"] > 0


class TestWalkForwardJobContract:
    """Test walkforward job artifact contracts."""

    def test_walkforward_produces_required_artifacts(
        self, job_id: str, sample_predictions_csv: Path
    ):
        """Walkforward job should produce results.csv, summary.csv, and result.json."""
        request = WalkForwardRequest(
            frequency="60min",
            symbol="NVDA",
            t_start="2024-01-01",
            t_end="2024-02-01",
            test_span_months=1,
            train_lookback_months=3,
            min_lookback_months=1,
            first_test_start=None,
            predictions_csv=str(sample_predictions_csv),
            parameter_sets=[
                {
                    "label": "baseline",
                    "k_sigma_long": 1.0,
                    "k_sigma_short": 1.0,
                    "k_atr_long": 2.0,
                    "k_atr_short": 2.0,
                    "risk_per_trade_pct": 1.0,
                    "reward_risk_ratio": 2.0,
                }
            ],
        )

        walkforward_job.run(job_id, request)

        job_dir = store.get_job_dir(job_id)
        assert (job_dir / "artifacts" / "results.csv").exists()
        assert (job_dir / "artifacts" / "summary.csv").exists()
        assert (job_dir / "result.json").exists()

    def test_walkforward_results_csv_schema(
        self, job_id: str, sample_predictions_csv: Path
    ):
        """Results CSV should have fold and parameter label columns."""
        request = WalkForwardRequest(
            frequency="60min",
            symbol="NVDA",
            t_start="2024-01-01",
            t_end="2024-02-01",
            test_span_months=1,
            train_lookback_months=3,
            min_lookback_months=1,
            first_test_start=None,
            predictions_csv=str(sample_predictions_csv),
            parameter_sets=[
                {
                    "label": "baseline",
                    "k_sigma_long": 1.0,
                    "k_sigma_short": 1.0,
                    "k_atr_long": 2.0,
                    "k_atr_short": 2.0,
                    "risk_per_trade_pct": 1.0,
                    "reward_risk_ratio": 2.0,
                }
            ],
        )

        walkforward_job.run(job_id, request)

        results_df = pd.read_csv(
            store.get_job_dir(job_id) / "artifacts" / "results.csv"
        )

        # Should have fold structure columns
        fold_cols = {
            "param_label",
            "fold_idx",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
        }
        assert fold_cols.issubset(set(results_df.columns))

        # Should have metrics
        metric_cols = {
            "sharpe_ratio",
            "total_return",
            "max_drawdown",
            "n_trades",
        }
        assert metric_cols.issubset(set(results_df.columns))

    def test_walkforward_summary_csv_schema(
        self, job_id: str, sample_predictions_csv: Path
    ):
        """Summary CSV should aggregate Sharpe statistics per parameter label."""
        request = WalkForwardRequest(
            frequency="60min",
            symbol="NVDA",
            t_start="2024-01-01",
            t_end="2024-02-01",
            test_span_months=1,
            train_lookback_months=3,
            min_lookback_months=1,
            first_test_start=None,
            predictions_csv=str(sample_predictions_csv),
            parameter_sets=[
                {
                    "label": "baseline",
                    "k_sigma_long": 1.0,
                    "k_sigma_short": 1.0,
                    "k_atr_long": 2.0,
                    "k_atr_short": 2.0,
                    "risk_per_trade_pct": 1.0,
                    "reward_risk_ratio": 2.0,
                }
            ],
        )

        walkforward_job.run(job_id, request)

        summary_df = pd.read_csv(
            store.get_job_dir(job_id) / "artifacts" / "summary.csv"
        )

        expected_cols = {
            "param_label",
            "mean_sharpe",
            "std_sharpe",
            "min_sharpe",
            "max_sharpe",
            "n_folds",
            "p_sharpe_gt_0",
            "robustness_score",
        }
        assert expected_cols.issubset(set(summary_df.columns))
        assert len(summary_df) == 1  # One row per parameter set
        assert summary_df["param_label"].iloc[0] == "baseline"


class TestTrainJobContract:
    """Test train job artifact contracts."""

    def test_train_produces_required_artifacts(self, job_id: str, tmp_path: Path):
        """Train job should produce model checkpoint, metrics, and predictions CSV."""
        # Note: This test would require more setup (data files, etc.)
        # For now, we document the expected artifacts
        pass  # Placeholder - train job tests require full data setup

    def test_train_metrics_json_schema(self):
        """Train metrics.json should have loss, val_loss, and training metadata."""
        # Expected schema:
        expected_keys = {
            "final_loss",
            "final_val_loss",
            "best_epoch",
            "n_epochs",
            "train_time_seconds",
        }
        # Actual validation would happen after running a real train job
        pass  # Placeholder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
