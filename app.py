import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# --- Page Configuration ---
st.set_page_config(
    page_title="Pollution Forecast Dashboard",
    page_icon="💨",
    layout="wide"
)

# --- Load Data and Model ---
# Load the model (this can take a moment)
@st.cache_resource
def load_keras_model():
    return load_model('pollution_model.h5')

model = load_keras_model()

# Load the saved test data and predictions
y_test_actual = np.load('y_test_actual.npy')
predictions = np.load('predictions.npy')

# Create a clean dataframe for plotting
plot_df = pd.DataFrame({
    'Actual PM2.5': y_test_actual.flatten(),
    'Predicted PM2.5': predictions.flatten()
})

# --- Dashboard UI ---
st.title("💨 B.Tech Project: Pollution Forecasting with LSTM")

# --- START: Added Project and Student Info ---
st.markdown("---") # Adds a horizontal line
col1, col2 = st.columns(2)

with col1:
    st.subheader("Student Details")
    st.markdown(
        """
        - **Name:** SATYAM GUPTA
        - **College ID:** 2023UCI8027
        - **Branch:** CSIOT
        """
    )

with col2:
    st.subheader("Project Details")
    st.markdown(
        """
        - **Project:** AI/ML Endsem Project
        - **Semester:** 5th Sem
        - **Year:** 2025-26
        """
    )
st.markdown("---") # Adds another horizontal line
# --- END: Added Project and Student Info ---


st.write("This dashboard shows the performance of our trained LSTM model on the test data.")

# --- 1. Overall Performance Plot ---
st.header("Overall Model Performance (on Test Data)")
st.write("This interactive chart shows the predicted values (red) vs. the actual values (blue) for the entire test dataset. You can zoom and pan!")

fig_all = go.Figure()
fig_all.add_trace(go.Scatter(
    y=plot_df['Actual PM2.5'],
    name='Actual (Ground Truth)',
    line=dict(color='blue', width=2)
))
fig_all.add_trace(go.Scatter(
    y=plot_df['Predicted PM2.5'],
    name='Predicted (Model)',
    line=dict(color='red', width=2, dash='dash')
))
fig_all.update_layout(
    xaxis_title="Time (Hours)",
    yaxis_title="PM2.5 (μg/m³)",
    legend_title="Legend"
)
st.plotly_chart(fig_all, use_container_width=True)

# --- 2. Simulated 24-Hour Forecast ---
st.header("Simulated 24-Hour Forecast")
st.write("Use the slider to select a 24-hour window from the test data. This simulates how the model would have performed on a given day.")

# Create a slider to select the starting hour
# We subtract 24 so the slider can select up to the last 24-hour block
max_start_hour = len(plot_df) - 24
start_hour = st.slider(
    "Select a start hour for the 24-hour forecast:",
    0, 
    max_start_hour,
    0  # Default value
)
end_hour = start_hour + 24

# Get the data for the selected 24-hour window
window_df = plot_df.iloc[start_hour:end_hour]

# Calculate RMSE for this specific window
window_rmse = np.sqrt(np.mean((window_df['Actual PM2.5'] - window_df['Predicted PM2.5'])**2))

st.metric(
    label=f"Forecast RMSE for this 24-Hour Window (Hours {start_hour}-{end_hour})",
    value=f"{window_rmse:.2f} μg/m³"
)

# Plot the 24-hour window
fig_24hr = go.Figure()
fig_24hr.add_trace(go.Scatter(
    x=window_df.index,
    y=window_df['Actual PM2.5'],
    name='Actual',
    line=dict(color='blue', width=3),
    mode='lines+markers'
))
fig_24hr.add_trace(go.Scatter(
    x=window_df.index,
    y=window_df['Predicted PM2.5'],
    name='Predicted',
    line=dict(color='red', width=3, dash='dash'),
    mode='lines+markers'
))
fig_24hr.update_layout(
    title=f"Forecast for Hours {start_hour} to {end_hour}",
    xaxis_title="Time (Hour in Window)",
    yaxis_title="PM2.5 (μg/m³)",
    legend_title="Legend"
)
st.plotly_chart(fig_24hr, use_container_width=True)

# --- 3. Generate a "Future" 24-Hour Forecast ---
# (I'm including the "Future Forecast" code we discussed, in case you wanted to add it)
st.header("Simulate a 'Future' 24-Hour Forecast")
st.write("This simulation shows how the model would predict the *next 24 hours* by feeding its own predictions back into itself (autoregression).")

# We need to load the other scalers and the last 24h of data
import joblib

# Constants from your Colab notebook
LOOKBACK = 24
N_FEATURES = 8 # (pm10, pm2_5, no2, so2, ozone, co, dust, aod)

# Load scalers
@st.cache_resource
def load_all_scalers():
    try:
        full_scaler = joblib.load('full_data_scaler.joblib')
        target_scaler = joblib.load('target_scaler.joblib')
        return full_scaler, target_scaler
    except FileNotFoundError:
        st.error("Scaler files not found! Make sure 'full_data_scaler.joblib' and 'target_scaler.joblib' are in the same folder.")
        return None, None

full_scaler, target_scaler = load_all_scalers()

# Load the raw data to get the starting point
@st.cache_data
def get_last_24_hours():
    try:
        # Load the original CSV, clean it, and get the last 24 hours
        df = pd.read_csv('open-meteo-22.50N77.80E375m.csv', skiprows=3)
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        df.columns = ['pm10', 'pm2_5', 'no2', 'so2', 'ozone', 'co2', 'co', 'dust', 'aod']
        df = df.drop('co2', axis=1)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Get the last 24 hours of the *entire* dataset
        last_24_hours_df = df.iloc[-LOOKBACK:]
        return last_24_hours_df.values # Return as a numpy array
    except FileNotFoundError:
        st.error("Original CSV 'open-meteo-22.50N77.80E375m.csv' not found.")
        return None

last_24_hours = get_last_24_hours()

# The "Forecast" button
if st.button("Generate Next 24-Hour Forecast"):
    if full_scaler and target_scaler and (last_24_hours is not None):
        # 1. Scale the input data
        current_batch_scaled = full_scaler.transform(last_24_hours)
        
        # 2. Reshape for the model
        current_batch_reshaped = np.reshape(current_batch_scaled, (1, LOOKBACK, N_FEATURES))
        
        forecast_list_scaled = []
        
        # ['pm10', 'pm2_5', 'no2', 'so2', 'ozone', 'co', 'dust', 'aod']
        target_idx = 1 # pm2_5 is the 2nd column, so index 1

        # 3. Loop 24 times to predict the next 24 hours
        for i in range(24):
            # Predict the next hour
            pred_scaled = model.predict(current_batch_reshaped)[0] # Get the single prediction [0.45]
            
            # Store this prediction
            forecast_list_scaled.append(pred_scaled)
            
            # --- Autoregressive step ---
            # Get all features from the last known time step
            new_row_features = current_batch_scaled[-1, :] 
            
            # Update the 'pm2_5' feature (index 1) with our new prediction
            new_row_features[target_idx] = pred_scaled[0]
            
            # Reshape this new row
            new_row_reshaped = new_row_features.reshape(1, N_FEATURES)
            
            # Create the new (1, 24, 8) batch
            # Get the old batch (all but the first row)
            new_batch_scaled = current_batch_scaled[1:, :]
            # Add our new_row to the end
            new_batch_scaled = np.append(new_batch_scaled, new_row_reshaped, axis=0)
            
            # Reshape for the next loop
            current_batch_reshaped = np.reshape(new_batch_scaled, (1, LOOKBACK, N_FEATURES))
            # Update current_batch_scaled as well
            current_batch_scaled = new_batch_scaled

        # 4. Inverse transform the predictions
        forecast_scaled = np.array(forecast_list_scaled)
        real_forecast = target_scaler.inverse_transform(forecast_scaled)

        # 5. Plot the forecast
        st.subheader("Forecast Results: Next 24 Hours")
        forecast_df = pd.DataFrame(real_forecast, columns=['Predicted PM2.5'])
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            y=forecast_df['Predicted PM2.5'],
            name='Forecasted PM2.5',
            line=dict(color='green', width=3, dash='dot'),
            mode='lines+markers'
        ))
        fig_forecast.update_layout(
            title="Simulated 'Future' Forecast",
            xaxis_title="Hour (from now)",
            yaxis_title="Predicted PM2.5 (μg/m³)",
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.error("Could not generate forecast. Check that all required files are present.")
