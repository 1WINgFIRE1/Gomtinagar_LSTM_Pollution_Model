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
        - **Branch:** CSIOT 5th sem
        """
    )

with col2:
    st.subheader("Project Details")
    st.markdown(
        """
        - **Project:** AI/ML Endsem Project (2025-26)
        - **Project Goal:** To Forecast PM2.5 of Gomtipura city (M.P)
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
