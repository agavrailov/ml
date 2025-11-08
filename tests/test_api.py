import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np
import pandas as pd

# Adjust the path to import from api.main
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import app
from src.config import TSTEPS

# Fixture for the test client (synchronous)
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# Test for the health check endpoint
def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# Test for successful prediction
@patch('api.main.predict_future_prices') # Corrected patch target
def test_predict_success(mock_predict_future_prices, client: TestClient):
    # Mock the prediction function to return a known value
    mock_predict_future_prices.return_value = 123.45

    # Create dummy OHLC data of correct length
    dummy_data = []
    for _ in range(TSTEPS):
        dummy_data.append({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5})
    
    response = client.post("/predict", json={"data": dummy_data})
    
    assert response.status_code == 200
    assert response.json() == {"predicted_price": 123.45}
    
    # Verify that predict_future_prices was called with the correct data
    assert mock_predict_future_prices.called
    call_args, _ = mock_predict_future_prices.call_args
    passed_df = call_args[0]
    assert isinstance(passed_df, pd.DataFrame)
    assert len(passed_df) == TSTEPS
    assert passed_df.iloc[0]['Open'] == 100.0

# Test for invalid input length
def test_predict_invalid_length(client: TestClient):
    # Create dummy OHLC data with incorrect length (e.g., TSTEPS - 1)
    dummy_data = []
    for _ in range(TSTEPS - 1):
        dummy_data.append({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5})
    
    response = client.post("/predict", json={"data": dummy_data})
    
    assert response.status_code == 400
    assert f"Input data must contain exactly {TSTEPS} OHLC data points." in response.json()["detail"]

# Test for invalid data format (missing field)
def test_predict_invalid_format(client: TestClient):
    dummy_data = []
    for _ in range(TSTEPS):
        dummy_data.append({"Open": 100.0, "High": 101.0, "Low": 99.0}) # Missing "Close"
    
    response = client.post("/predict", json={"data": dummy_data})
    
    assert response.status_code == 422 # Unprocessable Entity due to Pydantic validation
    assert "Field required" in response.json()["detail"][0]["msg"] # Corrected case-sensitivity

# Test for internal server error during prediction
@patch('api.main.predict_future_prices') # Corrected patch target
def test_predict_internal_error(mock_predict_future_prices, client: TestClient):
    # Mock the prediction function to raise an exception
    mock_predict_future_prices.side_effect = Exception("Model prediction failed")

    dummy_data = []
    for _ in range(TSTEPS):
        dummy_data.append({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5})
    
    response = client.post("/predict", json={"data": dummy_data})
    
    assert response.status_code == 500
    assert "Prediction failed: Model prediction failed" in response.json()["detail"]