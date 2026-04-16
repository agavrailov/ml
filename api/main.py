from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import json
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.predict import predict_future_prices
from src.config import TSTEPS, N_FEATURES, get_active_model_path, BASE_DIR

app = FastAPI(
    title="ML LSTM Price Prediction API",
    description="API for predicting future prices using a trained LSTM model.",
    version="1.0.0",
)

# Pydantic model for a single OHLC data point
class OHLCDataPoint(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float

# Pydantic model for the prediction request body
class PredictionRequest(BaseModel):
    data: List[OHLCDataPoint]

@app.post("/predict", summary="Predict future price", response_description="The predicted future price and model version")
async def predict(request: PredictionRequest):
    """
    Accepts a list of historical OHLC data points and returns a future price prediction.

    The input `data` list must contain enough data points to allow for feature
    engineering and still provide `TSTEPS` (configured in src/config.py) data points
    for the model's input. This means `20 + TSTEPS` OHLC data points are required.
    """
    required_data_points = 20 + TSTEPS # 20 for SMA_21, plus TSTEPS for the model input
    if len(request.data) < required_data_points:
        raise HTTPException(
            status_code=400,
            detail=f"Input data must contain at least {required_data_points} OHLC data points to allow for feature engineering and {TSTEPS} timesteps for the model. (Currently, TSTEPS={TSTEPS})"
        )

    # Convert list of Pydantic models to a Pandas DataFrame
    # The order of columns is important for the model
    input_df = pd.DataFrame([item.model_dump() for item in request.data])
    
    # Ensure column order matches expected N_FEATURES (OHLC)
    expected_columns = ['Open', 'High', 'Low', 'Close']
    if not all(col in input_df.columns for col in expected_columns):
        raise HTTPException(
            status_code=400,
            detail="Input data must contain 'Open', 'High', 'Low', 'Close' columns."
        )
    input_df = input_df[expected_columns]

    try:
        # Pass the full input_df to predict_future_prices, which will handle tailing TSTEPS
        predicted_price = predict_future_prices(input_df)
        
        # Get the active model path and extract the timestamp for versioning
        active_model_path = get_active_model_path()
        if active_model_path is None:
            raise HTTPException(status_code=500, detail="No active model found. Please train and promote a model first.")
        
        model_version = os.path.basename(active_model_path).replace('my_lstm_model_', '').replace('.keras', '')
        
        return {"predicted_price": predicted_price, "model_version": model_version}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health", summary="Health check", response_description="API health status")
async def health_check():
    """
    Checks the health of the API.
    """
    return {"status": "ok"}


_FREQUENCY_MINUTES = {
    "1min": 1, "5min": 5, "15min": 15, "30min": 30, "60min": 60, "1h": 60,
    "120min": 120, "2h": 120, "240min": 240, "4h": 240,
}

_STATUS_JSON_PATH = Path(BASE_DIR) / "ui_state" / "live" / "status.json"


@app.get("/health/trading", summary="Live trading health", response_description="Trading daemon health status")
async def trading_health():
    """Return daemon health: last bar age, system state, position status, alert count.

    HTTP 200 — healthy.
    HTTP 503 — daemon is stale (no bar for >2x bar frequency) or in FATAL_ERROR state.
    HTTP 404 — status file not found (daemon has never started).
    """
    if not _STATUS_JSON_PATH.exists():
        raise HTTPException(status_code=404, detail="Trading daemon has not started (status.json not found)")

    try:
        raw = json.loads(_STATUS_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read status.json: {exc}")

    # Compute last bar age
    last_bar_time_str: str | None = (raw.get("last_bar_info") or {}).get("time")
    last_bar_age_minutes: float | None = None
    if last_bar_time_str:
        try:
            last_bar_dt = datetime.fromisoformat(last_bar_time_str)
            if last_bar_dt.tzinfo is None:
                last_bar_dt = last_bar_dt.replace(tzinfo=timezone.utc)
            last_bar_age_minutes = round(
                (datetime.now(timezone.utc) - last_bar_dt).total_seconds() / 60, 1
            )
        except Exception:
            pass

    state: str = raw.get("state", "UNKNOWN")
    has_open_position: bool = (raw.get("position_info") or {}).get("status") == "OPEN"
    alert_count: int = raw.get("alert_count", 0)
    frequency: str = raw.get("frequency", "60min")
    bar_freq_minutes = _FREQUENCY_MINUTES.get(frequency, 60)

    payload = {
        "state": state,
        "last_bar_age_minutes": last_bar_age_minutes,
        "has_open_position": has_open_position,
        "alert_count": alert_count,
        "bar_frequency_minutes": bar_freq_minutes,
    }

    stale = last_bar_age_minutes is not None and last_bar_age_minutes > 2 * bar_freq_minutes
    fatal = state == "FATAL_ERROR"
    if stale or fatal:
        return JSONResponse(status_code=503, content=payload)

    return payload


# To run this API:
# uvicorn api.main:app --reload --port 8000
# Then access http://127.0.0.1:8000/docs for Swagger UI
