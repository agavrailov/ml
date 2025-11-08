import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import altair as alt
import os
import sys

# Add the project root to the Python path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import TSTEPS

# --- Configuration ---
API_URL = "http://localhost:8000/predict" # Assuming FastAPI runs on port 8000

st.set_page_config(layout="wide")

st.title("📈 ML LSTM Price Prediction UI")
st.markdown("Enter historical OHLC data to get a future price prediction.")

# --- Data Input ---
st.header("1. Input Historical Data")

input_method = st.radio("Choose input method:", ("Manual Input", "Upload CSV"))

ohlc_data = []

if input_method == "Manual Input":
    st.info(f"Please enter exactly {TSTEPS} rows of OHLC data.")
    
    # Create a DataFrame to hold manual input
    manual_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close'])
    
    # Use st.data_editor for easy input
    edited_df = st.data_editor(
        manual_df,
        num_rows="dynamic",
        column_config={
            "Open": st.column_config.NumberColumn(format="%.2f"),
            "High": st.column_config.NumberColumn(format="%.2f"),
            "Low": st.column_config.NumberColumn(format="%.2f"),
            "Close": st.column_config.NumberColumn(format="%.2f"),
        },
        height=200 + TSTEPS * 35, # Adjust height based on TSTEPS
        key="manual_input_editor"
    )
    
    if len(edited_df) > 0:
        ohlc_data = edited_df.to_dict(orient="records")

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(df_uploaded)

        if len(df_uploaded) != TSTEPS:
            st.error(f"Uploaded CSV must contain exactly {TSTEPS} rows of OHLC data.")
        else:
            # Ensure required columns are present
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df_uploaded.columns for col in required_cols):
                st.error(f"Uploaded CSV must contain columns: {', '.join(required_cols)}")
            else:
                ohlc_data = df_uploaded[required_cols].to_dict(orient="records")

# --- Prediction Button ---
st.header("2. Get Prediction")

if st.button("Predict Future Price"):
    if not ohlc_data:
        st.error("Please input historical data first.")
    elif len(ohlc_data) != TSTEPS:
        st.error(f"Input data must contain exactly {TSTEPS} OHLC data points for prediction.")
    else:
        with st.spinner("Making prediction..."):
            try:
                # Prepare data for API request
                request_data = {"data": ohlc_data}
                
                # Make API call
                response = requests.post(API_URL, json=request_data)
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
                
                prediction_result = response.json()
                predicted_price = prediction_result.get("predicted_price")

                if predicted_price is not None:
                    st.success(f"Predicted Future Price: **{predicted_price:.2f}**")

                    # --- Visualization ---
                    st.header("3. Visualization")
                    
                    # Convert input data to DataFrame for plotting
                    plot_df = pd.DataFrame(ohlc_data)
                    plot_df['index'] = range(len(plot_df)) # Add an index for plotting
                    
                    # Add the predicted price as a new point
                    # Assuming prediction is for the next step after the last input data point
                    predicted_point = pd.DataFrame([{
                        'index': len(plot_df),
                        'Open': predicted_price, # Use predicted price for 'Open' for simplicity in plotting
                        'High': predicted_price,
                        'Low': predicted_price,
                        'Close': predicted_price,
                        'type': 'Prediction'
                    }])
                    plot_df['type'] = 'Historical'
                    plot_df_combined = pd.concat([plot_df, predicted_point], ignore_index=True)

                    # Create a base chart
                    base = alt.Chart(plot_df_combined).encode(
                        x=alt.X('index', title='Time Step')
                    )

                    # Historical prices (candlestick or line)
                    # For simplicity, let's plot 'Close' prices as a line
                    historical_chart = base.mark_line(point=True, color='blue').encode(
                        y=alt.Y('Close', title='Price'),
                        tooltip=['Open', 'High', 'Low', 'Close']
                    ).transform_filter(
                        alt.FieldEqualPredicate(field="type", equal="Historical")
                    )

                    # Predicted price point
                    prediction_chart = base.mark_point(color='red', size=100, filled=True, shape="diamond").encode(
                        y=alt.Y('Open', title='Price'), # Using 'Open' for predicted_price
                        tooltip=['Open']
                    ).transform_filter(
                        alt.FieldEqualPredicate(field="type", equal="Prediction")
                    )

                    # Combine charts
                    chart = (historical_chart + prediction_chart).properties(
                        title="Historical Data and Predicted Price"
                    ).interactive()
                    
                    st.altair_chart(chart, use_container_width=True)

                else:
                    st.error("Prediction API did not return a predicted price.")

            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to the Prediction API at {API_URL}. Please ensure the API is running.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API Error: {e.response.status_code} - {e.response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --- Instructions to run API ---
st.sidebar.header("API Instructions")
st.sidebar.info("To run the Prediction API, execute `uvicorn api.main:app --reload --port 8000` in your terminal.") # Simplified