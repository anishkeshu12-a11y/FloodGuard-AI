import streamlit as st
import pandas as pd
import pydeck as pdk
import xgboost as xgb
import os

st.set_page_config(page_title="FloodGuard AI", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model('models/xgboost_flood_model.json')
    return model

@st.cache_data
def load_grid():
    return pd.read_csv('data/grid_data.csv')

st.title("🌊 FloodGuard: AI-Powered Flood Risk Prediction & Early Warning System")
st.markdown("Predict urban flood risk levels in real-time based on live weather data using **XGBoost** and Geospatial data.")

with st.spinner('Loading Models & Geospatial Data...'):
    model = load_model()
    df = load_grid()

# Sidebar
st.sidebar.header("🌦️ Weather Simulation")
st.sidebar.markdown("Adjust the synthetic real-time weather readings for the monitored region.")
rainfall = st.sidebar.slider("Current Rainfall (mm)", 0, 100, 20)
rainfall_lag_1 = st.sidebar.slider("Rainfall Yesterday (mm)", 0, 100, 30)
rainfall_lag_2 = st.sidebar.slider("Rainfall 2 Days Ago (mm)", 0, 100, 10)

# Predict
input_data = df[['elevation', 'slope']].copy()
input_data['rainfall'] = rainfall
input_data['rainfall_lag_1'] = rainfall_lag_1
input_data['rainfall_lag_2'] = rainfall_lag_2

# Feature order must match training
feature_cols = ['elevation', 'slope', 'rainfall', 'rainfall_lag_1', 'rainfall_lag_2']
predictions = model.predict(input_data[feature_cols])

df['risk'] = predictions

# Colors
def get_color(risk):
    # RGB
    if risk == 0:
        return [0, 255, 0, 150]    # Low: Green
    elif risk == 1:
        return [255, 165, 0, 150]  # Medium: Orange
    else:
        return [255, 0, 0, 150]    # High: Red

df['color'] = df['risk'].apply(get_color)

# Metrics Summary
st.markdown("### Risk Snapshot")
col1, col2, col3 = st.columns(3)
col1.metric("Low Risk Zones", len(df[df['risk'] == 0]))
col2.metric("Medium Risk Zones", len(df[df['risk'] == 1]))
col3.metric("High Risk Zones", len(df[df['risk'] == 2]))

# PyDeck Map
st.subheader("Interactive Real-Time Flood Risk Map")

layer = pdk.Layer(
    "ColumnLayer",
    data=df,
    get_position=['longitude', 'latitude'],
    get_elevation='elevation',
    elevation_scale=50, # scale up the columns visually
    radius=150,
    get_fill_color='color',
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    longitude=df['longitude'].mean(),
    latitude=df['latitude'].mean(),
    zoom=10,
    pitch=45,
    bearing=0
)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Elevation: {elevation}m\nRisk Class: {risk}"}
)

st.pydeck_chart(r)

st.markdown("""
**Risk Levels:**
- 🟢 **Low Risk**: Regular conditions, no immediate action required.
- 🟠 **Medium Risk**: Prepare for potential waterlogging in low-lying areas.
- 🔴 **High Risk**: Immediate flooding imminent. Evacuate vulnerable zones.
""")
