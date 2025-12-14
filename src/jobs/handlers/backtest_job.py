from __future__ import annotations

import json

from src.backtest import run_backtest_for_ui
from src.core.contracts import BacktestRequest, BacktestResult
from src.jobs import store


def run(job_id: str, request: BacktestRequest) -> BacktestResult:
    """Execute a backtest job and write standard artifacts.

    Artifacts:
    - artifacts/equity.csv
    - artifacts/trades.csv
    - artifacts/metrics.json
    """

    equity_df, trades_df, metrics = run_backtest_for_ui(
        frequency=request.frequency,
        prediction_mode=request.prediction_mode,
        start_date=request.start_date,
        end_date=request.end_date,
        predictions_csv=request.predictions_csv,
        risk_per_trade_pct=request.risk_per_trade_pct,
        reward_risk_ratio=request.reward_risk_ratio,
        k_sigma_long=request.k_sigma_long,
        k_sigma_short=request.k_sigma_short,
        k_atr_long=request.k_atr_long,
        k_atr_short=request.k_atr_short,
        enable_longs=request.enable_longs,
        allow_shorts=request.allow_shorts,
    )

    equity_csv = store.write_df_csv_artifact(job_id, "equity.csv", equity_df)
    trades_csv = store.write_df_csv_artifact(job_id, "trades.csv", trades_df)
    store.write_text_artifact(
        job_id,
        "metrics.json",
        json.dumps(metrics, indent=2, ensure_ascii=False),
    )

    # Result.json should remain small and stable.
    result = BacktestResult(metrics=metrics, equity_csv=equity_csv, trades_csv=trades_csv)
    store.write_result(job_id, result.to_dict())

    return result
