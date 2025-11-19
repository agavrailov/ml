from pathlib import Path

import nbformat as nbf


def main() -> None:
    root = Path(__file__).resolve().parents[1]  # repo root = ml_lstm
    nb_path = root / "notebooks" / "workflow_cli_ingestion_to_papertrading.ipynb"
    nb_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()

    # 0) Global config cell
    global_cfg = nbf.v4.new_code_cell(
        source=(
            "# Global knobs for the workflow\n"
            "FREQUENCY = \"60min\"  # e.g. '15min', '30min', '60min'\n"
            "TSTEPS = 64          # must be consistent with your config / trained models\n"
            "\n"
            "# Paths derived from FREQUENCY that are reused across steps\n"
            "PREDICTIONS_CSV = f\"backtests/nvda_{FREQUENCY}_predictions.csv\"\n"
            "TRADES_CSV      = f\"backtests/nvda_{FREQUENCY}_trades.csv\"\n"
            "EQUITY_CSV      = f\"backtests/nvda_{FREQUENCY}_equity.csv\"\n"
            "PRICE_CSV       = f\"data/processed/nvda_{FREQUENCY}.csv\"\n"
        )
    )

    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# ml_lstm: End-to-end CLI workflow (ingestion → paper trading)\n\n"
            "This notebook collects the major project CLIs needed to run the full "
            "workflow from data ingestion to paper trading.\n\n"
            "High-level steps:\n"
            "1. Run the daily data pipeline (ingestion, cleaning, gap handling, feature engineering).\n"
            "2. Train the LSTM model.\n"
            "3. Evaluate the latest best model.\n"
            "4. Generate per-bar predictions CSV.\n"
            "5. Run a backtest using that CSV.\n"
            "6. Plot backtest diagnostics.\n"
            "7. Run simulated paper trading using the same CSV.\n\n"
            "All `!` shell commands assume this notebook is in `notebooks/` and the "
            "project root is the parent directory."
        ),
        global_cfg,
        nbf.v4.new_markdown_cell(
            "## 1. Daily data pipeline (ingestion → processed hourly features)\n\n"
            "Runs `src.daily_data_agent` which orchestrates ingestion, cleaning, gap handling, "
            "curated minute snapshot, and resampling + feature engineering."
        ),
        nbf.v4.new_code_cell(
            "!cd .. && python -m src.daily_data_agent\n"
            "# For a dry run that skips IB/TWS ingestion, use:\n"
            "# !cd .. && python -m src.daily_data_agent --skip-ingestion\n"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Train LSTM model\n\n"
            "Train an LSTM model for the chosen `FREQUENCY` and `TSTEPS` using `src.train`."
        ),
        nbf.v4.new_code_cell(
            "!cd .. && python -m src.train --frequency {FREQUENCY} --tsteps {TSTEPS}\n"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Evaluate latest best model\n\n"
            "Evaluate the latest best model (based on `best_hyperparameters.json`) using `src.evaluate_model`."
        ),
        nbf.v4.new_code_cell(
            "!cd .. && python -m src.evaluate_model\n"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Generate per-bar predictions CSV\n\n"
            "Use `scripts.generate_predictions_csv` to create a per-bar predictions file for NVDA "
            "at the chosen `FREQUENCY`.\n\n"
            "Output columns: `Time`, `predicted_price`."
        ),
        nbf.v4.new_code_cell(
            "!cd .. && python -m scripts.generate_predictions_csv "
            "--frequency {FREQUENCY} --output {PREDICTIONS_CSV}\n"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Run backtest using predictions CSV\n\n"
            "Run `src.backtest` over processed OHLC data, using the predictions CSV as the prediction source.\n\n"
            "- `--prediction-mode csv` tells the backtest to read `predicted_price` from the CSV.\n"
            "- Exports trades and equity curve CSVs for diagnostics."
        ),
        nbf.v4.new_code_cell(
            "!cd .. && python -m src.backtest "
            "--frequency {FREQUENCY} "
            "--prediction-mode csv "
            "--predictions-csv {PREDICTIONS_CSV} "
            "--export-trades-csv {TRADES_CSV} "
            "--export-equity-csv {EQUITY_CSV}\n"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Plot backtest diagnostics\n\n"
            "Use `scripts.plot_backtest_diagnostics` to visualize equity curve and trade entry density."
        ),
        nbf.v4.new_code_cell(
            "!cd .. && python -m scripts.plot_backtest_diagnostics "
            "--equity {EQUITY_CSV} "
            "--trades {TRADES_CSV} "
            "--price-csv {PRICE_CSV}\n"
        ),
        nbf.v4.new_markdown_cell(
            "## 7. Paper trading over historical data\n\n"
            "Use `src.paper_trade` to run a simulated paper-trading session over historical NVDA data, "
            "reusing the predictions CSV."
        ),
        nbf.v4.new_code_cell(
            "!cd .. && python -m src.paper_trade "
            "--frequency {FREQUENCY} "
            "--predictions-csv {PREDICTIONS_CSV}\n"
        ),
        nbf.v4.new_markdown_cell(
            "## 8. (Optional) Hyperparameter tuning and experiment grid\n\n"
            "Not required for the minimal ingestion → paper-trading loop, but useful for experimentation."
        ),
        nbf.v4.new_code_cell(
            "# Optional: run hyperparameter tuning (can be long-running)\n"
            "# !cd .. && python -m src.tune_hyperparameters\n\n"
            "# Optional: run a grid of experiments (very long-running)\n"
            "# !cd .. && python -m src.experiment_runner\n"
        ),
    ]

    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }

    nbf.write(nb, nb_path)
    print(f"Notebook written to {nb_path}")


if __name__ == "__main__":
    main()