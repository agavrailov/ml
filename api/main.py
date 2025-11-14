from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import os
import sys

from src.predict import predict_future_prices
from src.config import TSTEPS, N_FEATURES, get_active_model_path # Import get_active_model_path

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

# To run this API:
# uvicorn api.main:app --reload --port 8000
# Then access http://127.0.0.1:8000/docs for Swagger UI
