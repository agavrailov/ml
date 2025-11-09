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

input_method = st.radio("Choose input method:", ("Manual Input", "Upload CSV", "Predict Latest (Auto-fetch)"))

ohlc_data = []
data_source_info = ""

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

elif input_method == "Predict Latest (Auto-fetch)":
    st.info(f"Automatically fetching the last {TSTEPS} rows from processed hourly data.")
    
    # More robust path construction
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    PROCESSED_HOURLY_DATA_CSV = os.path.join(project_root, 'data', 'processed', 'nvda_hourly.csv')
    st.write(f"Attempting to read from: {PROCESSED_HOURLY_DATA_CSV}") # Debugging line
    
    if os.path.exists(PROCESSED_HOURLY_DATA_CSV):
        try:
            df_hourly = pd.read_csv(PROCESSED_HOURLY_DATA_CSV)
            # st.write(f"DataFrame loaded. Shape: {df_hourly.shape}")
            # st.write(f"DataFrame columns: {df_hourly.columns.tolist()}")

            if 'Time' in df_hourly.columns: # Changed from 'DateTime' to 'Time'
                # st.write("Time column found. Attempting conversion to datetime.")
                df_hourly['DateTime'] = pd.to_datetime(df_hourly['Time']) # Create 'DateTime' column from 'Time'
                # st.write(f"DateTime conversion successful. First DateTime: {df_hourly['DateTime'].min()}")
                df_hourly = df_hourly.sort_values('DateTime').reset_index(drop=True)
            else:
                st.error("Error: 'Time' column not found in the processed hourly data.") # Changed error message
                st.stop() # Exit early if Time column is missing
            
            if len(df_hourly) >= TSTEPS:
                latest_data = df_hourly.tail(TSTEPS)
                ohlc_data = latest_data[['Open', 'High', 'Low', 'Close']].to_dict(orient="records")
                
                start_time = latest_data['DateTime'].min().strftime('%Y-%m-%d %H:%M')
                end_time = latest_data['DateTime'].max().strftime('%Y-%m-%d %H:%M')
                data_source_info = f"Data from {start_time} to {end_time}"
                
                # st.write("Data used for prediction:")
                st.dataframe(latest_data)
            else:
                st.warning(f"Not enough data in '{PROCESSED_HOURLY_DATA_CSV}' to fetch {TSTEPS} rows. Found {len(df_hourly)} rows.")
        except Exception as e:
            st.error(f"Error reading processed hourly data: {e}")
    else:
        st.warning(f"Processed hourly data file not found: '{PROCESSED_HOURLY_DATA_CSV}'. Please run data processing first.")

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
                model_version = prediction_result.get("model_version", "N/A")

                if predicted_price is not None:
                    st.success(f"Predicted Future Price: **{predicted_price:.2f}**")
                    st.info(f"Using Model Version: `{model_version}`")
                    if data_source_info:
                        st.info(f"Input Data Source: {data_source_info}")

                    # --- Visualization ---
                    st.header("3. Visualization")
                    
                    # Convert input data to DataFrame for plotting
                    # Ensure 'DateTime' is included if available from auto-fetch
                    if input_method == "Predict Latest (Auto-fetch)" and 'DateTime' in latest_data.columns:
                        plot_df = latest_data[['DateTime', 'Open', 'High', 'Low', 'Close']].copy()
                    else:
                        plot_df = pd.DataFrame(ohlc_data)
                        # For manual/CSV input, create a dummy DateTime for plotting
                        plot_df['DateTime'] = pd.to_datetime(pd.Series(range(len(plot_df))), unit='h', origin=pd.Timestamp('2023-01-01'))
                    
                    # Add the predicted price as a new point
                    # Assuming prediction is for the next step after the last input data point
                    predicted_point_datetime = plot_df['DateTime'].iloc[-1] + pd.Timedelta(hours=1)
                    predicted_point_df = pd.DataFrame([{
                        'DateTime': predicted_point_datetime,
                        'Open': predicted_price, # Use predicted price for 'Open' for simplicity in plotting
                        'High': predicted_price,
                        'Low': predicted_price,
                        'Close': predicted_price,
                        'type': 'Prediction'
                    }])
                    
                    plot_df['type'] = 'Historical'
                    plot_df_combined = pd.concat([plot_df, predicted_point_df], ignore_index=True)

                    # Filter to show only the last 200 items + the prediction
                    display_df = plot_df_combined.tail(201) # 200 historical + 1 prediction

                    # Create a base chart
                    base = alt.Chart(display_df).encode(
                        x=alt.X('DateTime:T', title='Date/Time')
                    )

                    # Candlestick chart for historical data
                    candlestick = base.mark_rule().encode(
                        y=alt.Y('Low', title='Price'),
                        y2='High',
                        color=alt.condition("datum.Open < datum.Close",
                                            alt.value("#06982d"), # Green
                                            alt.value("#dc3912")) # Red
                    ).transform_filter(
                        alt.FieldEqualPredicate(field="type", equal="Historical")
                    ) + base.mark_bar().encode(
                        y='Open',
                        y2='Close',
                        color=alt.condition("datum.Open < datum.Close",
                                            alt.value("#06982d"), # Green
                                            alt.value("#dc3912")) # Red
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
                    chart = (candlestick + prediction_chart).properties(
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