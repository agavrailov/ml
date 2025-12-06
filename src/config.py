# src/config.py

import os
import json
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import List

# --- Base paths ---
# Project root can be overridden if needed (e.g. for tests or deployment)
BASE_DIR = os.path.abspath(os.getenv("ML_LSTM_BASE_DIR", os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------
# Structured configuration
# ---------------------------

@dataclass(frozen=True)
class PathsConfig:
    """Filesystem and path configuration.

    Values can be overridden via environment variables:
    - ML_LSTM_RAW_DATA_DIR
    - ML_LSTM_PROCESSED_DATA_DIR
    - ML_LSTM_MODEL_SAVE_PATH
    - ML_LSTM_MODEL_REGISTRY_DIR
    - ML_LSTM_ACTIVE_MODEL_PATH_FILE
    """

    raw_data_dir: str = field(
        default_factory=lambda: os.getenv(
            "ML_LSTM_RAW_DATA_DIR", os.path.join(BASE_DIR, "data", "raw")
        )
    )
    processed_data_dir: str = field(
        default_factory=lambda: os.getenv(
            "ML_LSTM_PROCESSED_DATA_DIR", os.path.join(BASE_DIR, "data", "processed")
        )
    )
    model_save_path: str = field(
        default_factory=lambda: os.getenv(
            "ML_LSTM_MODEL_SAVE_PATH", os.path.join(BASE_DIR, "models", "my_lstm_model.keras")
        )
    )
    model_registry_dir: str = field(
        default_factory=lambda: os.getenv(
            "ML_LSTM_MODEL_REGISTRY_DIR", os.path.join(BASE_DIR, "models", "registry")
        )
    )
    active_model_path_file: str = field(
        default_factory=lambda: os.getenv(
            "ML_LSTM_ACTIVE_MODEL_PATH_FILE", os.path.join(BASE_DIR, "models", "active_model.txt")
        )
    )

    def raw_data_csv(self) -> str:
        return os.path.join(self.raw_data_dir, "nvda_minute.csv")

    def hourly_data_csv(self, frequency: str) -> str:
        return os.path.join(self.processed_data_dir, f"nvda_{frequency}.csv")

    def training_data_csv(self, frequency: str) -> str:
        return os.path.join(self.processed_data_dir, f"training_data_{frequency}.csv")

    def scaler_params_json(self, frequency: str) -> str:
        return os.path.join(self.processed_data_dir, f"scaler_params_{frequency}.json")

    def predictions_csv(self, symbol: str, frequency: str) -> str:
        return os.path.join(BASE_DIR, "backtests", f"{symbol.lower()}_{frequency}_predictions.csv")

    def trades_csv(self, symbol: str, frequency: str) -> str:
        return os.path.join(BASE_DIR, "backtests", f"{symbol.lower()}_{frequency}_trades.csv")

    def equity_csv(self, symbol: str, frequency: str) -> str:
        return os.path.join(BASE_DIR, "backtests", f"{symbol.lower()}_{frequency}_equity.csv")



@dataclass(frozen=True)
class TrainingConfig:
    """Model and training hyperparameters (single-run defaults + search spaces)."""

    # Default values for a single run
    frequency: str = "15min"
    tsteps: int = 5
    rows_ahead: int = 1
    tr_split: float = 0.7
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.003
    lstm_units: int = 64
    dropout_rate_1: float = 0.1
    dropout_rate_2: float = 0.1
    n_lstm_layers: int = 1
    stateful: bool = True
    n_features: int = 7  # Open, High, Low, Close, SMA_7, SMA_21, RSI
    optimizer_name: str = "rmsprop"
    loss_function: str = "mse"

    # Search spaces / options
    resample_frequencies: List[str] = field(default_factory=lambda: ["15min", "30min", "60min", "240min"])
    tsteps_options: List[int] = field(default_factory=lambda: [5, 8, 16, 24, 48])
    lstm_units_options: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    batch_size_options: List[int] = field(default_factory=lambda: [64, 128, 256])
    dropout_rate_options: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])
    n_lstm_layers_options: List[int] = field(default_factory=lambda: [1, 2])
    stateful_options: List[bool] = field(default_factory=lambda: [True, False])
    optimizer_options: List[str] = field(default_factory=lambda: ["rmsprop", "adam"])
    loss_function_options: List[str] = field(default_factory=lambda: ["mae", "mse"])
    features_to_use_options: List[List[str]] = field(
        default_factory=lambda: [
            ["Open", "High", "Low", "Close", "SMA_7", "SMA_21", "RSI"],
            ["Open", "High", "Low", "Close", "SMA_7", "SMA_21", "RSI", "Hour", "DayOfWeek"],
        ]
    )


@dataclass(frozen=True)
class ModelConfig:
    """Concrete model configuration used by the unified model API.

    This is a thin wrapper around :class:`TrainingConfig` and selected path
    settings. It is intentionally small and focused on the parameters actually
    required to build, train, and load the LSTM.
    """

    frequency: str
    tsteps: int
    n_features: int
    lstm_units: int
    batch_size: int
    learning_rate: float
    n_lstm_layers: int
    stateful: bool
    optimizer_name: str
    loss_function: str
    model_registry_dir: str


def get_model_config(
    *,
    frequency: str | None = None,
    tsteps: int | None = None,
    n_features: int | None = None,
    lstm_units: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    n_lstm_layers: int | None = None,
    stateful: bool | None = None,
    optimizer_name: str | None = None,
    loss_function: str | None = None,
) -> "ModelConfig":
    """Return an immutable model configuration for a training/prediction run.

    Defaults come from ``TRAINING`` and path aliases defined in this module.
    Callers can optionally override individual fields without having to know
    about all internal config globals.
    """

    base = TRAINING

    return ModelConfig(
        frequency=frequency or base.frequency,
        tsteps=tsteps or base.tsteps,
        n_features=n_features or base.n_features,
        lstm_units=lstm_units or base.lstm_units,
        batch_size=batch_size or base.batch_size,
        learning_rate=learning_rate or base.learning_rate,
        n_lstm_layers=n_lstm_layers or base.n_lstm_layers,
        stateful=base.stateful if stateful is None else stateful,
        optimizer_name=optimizer_name or base.optimizer_name,
        loss_function=loss_function or base.loss_function,
        model_registry_dir=MODEL_REGISTRY_DIR,
    )


@dataclass(frozen=True)
class IbConfig:
    """IB/TWS connection and data ingestion settings."""

    host: str = field(default_factory=lambda: os.getenv("TWS_HOST", "127.0.0.1"))
    port: int = int(os.getenv("TWS_PORT", "7496"))
    client_id: int = int(os.getenv("TWS_CLIENT_ID", "1"))
    max_concurrent_requests: int = 50
    data_batch_save_size: int = 7
    contract_details: dict = field(
        default_factory=lambda: {
            "symbol": "NVDA",
            "secType": "STK",
            "exchange": "SMART",
            "currency": "USD",
        }
    )


@dataclass(frozen=True)
class MarketConfig:
    """Market hours and calendar configuration."""

    timezone: str = "America/New_York"
    open_time: time = time(9, 30)
    close_time: time = time(16, 0)
    exchange_calendar_name: str = "XNYS"  # New York Stock Exchange


@dataclass(frozen=True)
class StrategyDefaultsConfig:
    """Defaults for trading strategy and backtest/paper-trade settings.

    Keeping these in one place avoids duplicating magic numbers in
    backtest/paper-trade code and makes it clear which knobs define the
    baseline trading behavior.
    """

    # Core risk and reward parameters
    risk_per_trade_pct: float = 0.02  # 1% of equity per trade
    reward_risk_ratio: float = 3.5

    # Noise / filter parameters
    # How much of the model residual sigma_err we subtract from the predicted move.
    k_sigma_err: float = 0.7
    # Minimum TP distance as a multiple of ATR.
    k_atr_min_tp: float = 0.7

    # Position sizing constraints
    min_position_size: float = 1.0

    # Execution / cost model
    commission_per_unit_per_leg: float = 0.005
    min_commission_per_order: float = 1.0
    # Account / session defaults
    initial_equity: float = 10_000.0


@dataclass(frozen=True)
class BacktestWindowConfig:
    """Global defaults for backtest/paper-trade date windows.

    ``default_start_date`` / ``default_end_date`` are ISO strings ("YYYY-MM-DD").
    When both are ``None``, the full available data range is used.
    """

    default_start_date: str | None = None
    default_end_date: str | None = None


# Instantiate structured configs
PATHS = PathsConfig()
TRAINING = TrainingConfig()
IB = IbConfig()
MARKET = MarketConfig()
STRATEGY_DEFAULTS = StrategyDefaultsConfig()
BACKTEST_WINDOW = BacktestWindowConfig()


# ---------------------------
# Tuned hyperparameter helpers
# ---------------------------

def load_tuned_hyperparameters(
    frequency: str,
    tsteps: int,
    best_hps_path: str = "best_hyperparameters.json",
) -> dict:
    """Return tuned hyperparameters for (frequency, tsteps), or an empty dict.

    The JSON is expected to have the form::

        {
            "15min": {
                "5": {
                    "validation_loss": ...,
                    "lstm_units": ...,
                    "batch_size": ...,
                    ...
                },
                ...
            },
            ...
        }
    """
    if not os.path.exists(best_hps_path):
        return {}

    try:
        with open(best_hps_path, "r") as f:
            # Handle empty or invalid JSON by falling back to defaults
            content = f.read().strip()
            if not content:
                return {}
            best_hps_data = json.loads(content)
    except json.JSONDecodeError:
        # If the file is corrupted or partially written, ignore it and use defaults
        return {}

    freq_block = best_hps_data.get(frequency, {})
    return freq_block.get(str(tsteps), {})


def get_run_hyperparameters(
    frequency: str | None = None,
    tsteps: int | None = None,
    best_hps_path: str = "best_hyperparameters.json",
) -> dict:
    """Return effective hyperparameters for a run.

    Start from TRAINING defaults and, if a tuned entry for (frequency, tsteps)
    exists in ``best_hyperparameters.json``, override the defaults with tuned
    values.
    """
    freq = frequency or TRAINING.frequency
    steps = tsteps or TRAINING.tsteps

    tuned = load_tuned_hyperparameters(freq, steps, best_hps_path=best_hps_path)

    return {
        "frequency": freq,
        "tsteps": steps,
        "lstm_units": tuned.get("lstm_units", TRAINING.lstm_units),
        "batch_size": tuned.get("batch_size", TRAINING.batch_size),
        "learning_rate": tuned.get("learning_rate", TRAINING.learning_rate),
        "epochs": tuned.get("epochs", TRAINING.epochs),
        "n_lstm_layers": tuned.get("n_lstm_layers", TRAINING.n_lstm_layers),
        "stateful": tuned.get("stateful", TRAINING.stateful),
        "optimizer_name": tuned.get("optimizer_name", TRAINING.optimizer_name),
        "loss_function": tuned.get("loss_function", TRAINING.loss_function),
        "features_to_use": tuned.get("features_to_use", TRAINING.features_to_use_options[0]),
    }


# ---------------------------
# Backwards-compatible aliases
# ---------------------------

# Model hyperparameters (single-run defaults)
FREQUENCY = TRAINING.frequency
TSTEPS = TRAINING.tsteps
ROWS_AHEAD = TRAINING.rows_ahead
TR_SPLIT = TRAINING.tr_split
BATCH_SIZE = TRAINING.batch_size
EPOCHS = TRAINING.epochs
LEARNING_RATE = TRAINING.learning_rate
LSTM_UNITS = TRAINING.lstm_units
DROPOUT_RATE_1 = TRAINING.dropout_rate_1
DROPOUT_RATE_2 = TRAINING.dropout_rate_2
N_LSTM_LAYERS = TRAINING.n_lstm_layers
STATEFUL = TRAINING.stateful
N_FEATURES = TRAINING.n_features
OPTIMIZER_NAME = TRAINING.optimizer_name
LOSS_FUNCTION = TRAINING.loss_function

# Hyperparameter options
RESAMPLE_FREQUENCIES = TRAINING.resample_frequencies
TSTEPS_OPTIONS = TRAINING.tsteps_options
LSTM_UNITS_OPTIONS = TRAINING.lstm_units_options
BATCH_SIZE_OPTIONS = TRAINING.batch_size_options
DROPOUT_RATE_OPTIONS = TRAINING.dropout_rate_options
N_LSTM_LAYERS_OPTIONS = TRAINING.n_lstm_layers_options
STATEFUL_OPTIONS = TRAINING.stateful_options
OPTIMIZER_OPTIONS = TRAINING.optimizer_options
LOSS_FUNCTION_OPTIONS = TRAINING.loss_function_options
FEATURES_TO_USE_OPTIONS = TRAINING.features_to_use_options

# Strategy / trading defaults (backwards-compatible flat aliases)
RISK_PER_TRADE_PCT = STRATEGY_DEFAULTS.risk_per_trade_pct
REWARD_RISK_RATIO = STRATEGY_DEFAULTS.reward_risk_ratio
K_SIGMA_ERR = STRATEGY_DEFAULTS.k_sigma_err
K_ATR_MIN_TP = STRATEGY_DEFAULTS.k_atr_min_tp
MIN_POSITION_SIZE = STRATEGY_DEFAULTS.min_position_size
COMMISSION_PER_UNIT_PER_LEG = STRATEGY_DEFAULTS.commission_per_unit_per_leg
MIN_COMMISSION_PER_ORDER = STRATEGY_DEFAULTS.min_commission_per_order
INITIAL_EQUITY = STRATEGY_DEFAULTS.initial_equity

# Backtest window defaults
BACKTEST_DEFAULT_START_DATE = BACKTEST_WINDOW.default_start_date
BACKTEST_DEFAULT_END_DATE = BACKTEST_WINDOW.default_end_date

# Paths
RAW_DATA_DIR = PATHS.raw_data_dir
PROCESSED_DATA_DIR = PATHS.processed_data_dir
MODEL_SAVE_PATH = PATHS.model_save_path
MODEL_REGISTRY_DIR = PATHS.model_registry_dir
ACTIVE_MODEL_PATH_FILE = PATHS.active_model_path_file  # legacy; kept for backwards-compatible fallback only
RAW_DATA_CSV = PATHS.raw_data_csv()

def get_hourly_data_csv_path(frequency, processed_data_dir=PROCESSED_DATA_DIR):
    """Generates the path for the resampled data CSV based on the given frequency."""
    return PATHS.hourly_data_csv(frequency)

def get_training_data_csv_path(frequency, processed_data_dir=PROCESSED_DATA_DIR):
    """Generates the path for the training data CSV based on the given frequency."""
    return PATHS.training_data_csv(frequency)

def get_scaler_params_json_path(frequency, processed_data_dir=PROCESSED_DATA_DIR):
    """Generates the path for the scaler parameters JSON based on the given frequency."""
    return PATHS.scaler_params_json(frequency)

def get_predictions_csv_path(symbol: str, frequency: str) -> str:
    """Path for predictions CSV in the backtests directory."""
    return PATHS.predictions_csv(symbol, frequency)

def get_trades_csv_path(symbol: str, frequency: str) -> str:
    """Path for trades CSV in the backtests directory."""
    return PATHS.trades_csv(symbol, frequency)

def get_equity_csv_path(symbol: str, frequency: str) -> str:
    """Path for equity CSV in the backtests directory."""
    return PATHS.equity_csv(symbol, frequency)

def get_active_model_path():
    """Resolve the path of the currently *active* model.

    The active model is defined as the globally best entry in
    ``best_hyperparameters.json`` (lowest validation_loss across all
    (frequency, tsteps) combinations). The legacy text pointer file is only
    used as a final fallback for backwards compatibility.
    """

    # 1) Global best from best_hyperparameters.json
    model_path, *_ = get_latest_best_model_path()
    if model_path:
        return model_path

    # 2) Legacy pointer file (text or JSON).
    if os.path.exists(ACTIVE_MODEL_PATH_FILE):
        try:
            with open(ACTIVE_MODEL_PATH_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except OSError:
            return None

        if not content:
            return None

        # Newer versions may store a small JSON object {"path": "..."}.
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "path" in data:
                return data["path"]
        except json.JSONDecodeError:
            # Treat the raw content as a plain filesystem path.
            return content

    return None

def get_latest_best_model_path(target_frequency=None, tsteps=None):
    """Resolve the best model for a given (frequency, tsteps).

    This consults ``best_hyperparameters.json`` **in the project root** (``BASE_DIR``)
    instead of the current working directory, so it behaves consistently when
    called from scripts, CLIs, or notebooks located in subdirectories.

    Returns a tuple ``(model_path, bias_correction_path, features_to_use_trained,
    lstm_units_trained, n_lstm_layers_trained)``. Any element can be ``None``
    if the corresponding information is not available.
    """
    best_hps_path = os.path.join(BASE_DIR, 'best_hyperparameters.json')
    if not os.path.exists(best_hps_path):
        return None, None, None, None, None

    # Handle empty or malformed JSON defensively so prediction code can
    # fall back to training defaults.
    with open(best_hps_path, 'r') as f:
        content = f.read().strip()
        if not content:
            return None, None, None, None, None
        try:
            best_hps_data = json.loads(content)
        except json.JSONDecodeError:
            return None, None, None, None, None

    best_loss = float('inf')
    best_model_filename = None
    best_bias_correction_filename = None
    features_to_use_trained = None
    lstm_units_trained = None
    n_lstm_layers_trained = None
    
    for freq, tsteps_data in best_hps_data.items():
        if target_frequency and freq != target_frequency:
            continue
        for tstep_val_str, metrics in tsteps_data.items():
            current_tsteps = int(tstep_val_str)
            if tsteps and current_tsteps != tsteps:
                continue
            
            if metrics['validation_loss'] < best_loss:
                best_loss = metrics['validation_loss']
                best_model_filename = metrics['model_filename']
                best_bias_correction_filename = metrics.get('bias_correction_filename')
                features_to_use_trained = metrics.get('features_to_use')
                lstm_units_trained = metrics.get('lstm_units')
                n_lstm_layers_trained = metrics.get('n_lstm_layers')
    
    model_path = None
    bias_correction_path = None

    if best_model_filename:
        model_path = os.path.join(MODEL_REGISTRY_DIR, best_model_filename)
    if best_bias_correction_filename:
        bias_correction_path = os.path.join(MODEL_REGISTRY_DIR, best_bias_correction_filename)

    return model_path, bias_correction_path, features_to_use_trained, lstm_units_trained, n_lstm_layers_trained

# --- Data Ingestion Configuration (backwards-compatible aliases) ---
TWS_HOST = IB.host
TWS_PORT = IB.port
TWS_CLIENT_ID = IB.client_id
TWS_MAX_CONCURRENT_REQUESTS = IB.max_concurrent_requests
DATA_BATCH_SAVE_SIZE = IB.data_batch_save_size

# Contract details for NVDA
NVDA_CONTRACT_DETAILS = IB.contract_details

# Market hours for gap analysis (New York time)
MARKET_TIMEZONE = MARKET.timezone
MARKET_OPEN_TIME = MARKET.open_time
MARKET_CLOSE_TIME = MARKET.close_time
EXCHANGE_CALENDAR_NAME = MARKET.exchange_calendar_name  # New York Stock Exchange
