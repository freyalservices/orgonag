import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from scipy.spatial import KDTree
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
from streamlit_autorefresh import st_autorefresh

# ========== CONFIG ==========
st.set_page_config(layout="wide")
st.title("🚨 Track Condition Dashboard with Live Location")

# ========== FILE LOADERS ==========
@st.cache_data
def load_dtn(subdivision):
    df = pd.read_csv("dataset/dtn.csv")
    return df[df['Subdivision'] == subdivision]

@st.cache_data
def load_tec(subdivision):
    df = pd.read_csv("dataset/tec.csv")
    return df[df['Subdivision'] == subdivision]

@st.cache_data
def load_grms_file(filepath):
    try:
        chunks = pd.read_csv(filepath, chunksize=50000)
        df = pd.concat(chunks)
        df.columns = [col.strip() for col in df.columns]
        required_cols = ["# Milepost", "Milepost Feet", "Latitude", "Longitude"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            return pd.DataFrame()
        df["# Milepost"] = pd.to_numeric(df["# Milepost"], errors="coerce")
        df["Milepost Feet"] = pd.to_numeric(df["Milepost Feet"], errors="coerce")
        df = df.dropna(subset=["# Milepost", "Milepost Feet", "Latitude", "Longitude"])
        df["MP"] = df["# Milepost"] + df["Milepost Feet"] / 5280
        if len(df) > 100000:
            df = df.iloc[::5]
        return df
    except Exception as e:
        st.error(f"Error loading GRMS file: {e}")
        return pd.DataFrame()

# ========== GLOBAL FILTERS ==========
st.subheader("Global Filters")
dtn_raw = pd.read_csv("dataset/dtn.csv")
subdivision = st.selectbox("Choose Subdivision", options=dtn_raw['Subdivision'].dropna().unique())
date_range = st.date_input("Optional: Date Range", [])

# ========== LOAD DATA ==========
dtn_df = load_dtn(subdivision)
tec_df = load_tec(subdivision)

# ========== SIDEBAR FILTERS ==========
st.sidebar.title("Refine Filters")
status_filter = st.sidebar.multiselect("DTN Status", options=dtn_df['Status'].dropna().unique())
severity_filter = st.sidebar.multiselect("TEC Severity", options=tec_df['Severity'].dropna().unique())
user_review_filter = st.sidebar.multiselect("User Review Status", options=tec_df['User Review Status'].dropna().unique())

if status_filter:
    dtn_df = dtn_df[dtn_df['Status'].isin(status_filter)]
if severity_filter:
    tec_df = tec_df[tec_df['Severity'].isin(severity_filter)]
if user_review_filter:
    tec_df = tec_df[tec_df['User Review Status'].isin(user_review_filter)]

# ========== LIVE TRACKING ==========
st.subheader("📍 Live Location Tracking")

track_enabled = st.checkbox("Enable Live Tracking", help="Allows browser to fetch your current location.")

if track_enabled:
    # Refresh every 5 seconds
    st_autorefresh(interval=5000, key="location_tracker")

    # Always get new location
    location = streamlit_geolocation()
    user_lat = location.get("latitude") if location else None
    user_lon = location.get("longitude") if location else None

    if user_lat and user_lon:
        st.success(f"Your current location: ({user_lat:.5f}, {user_lon:.5f})")
    else:
        st.warning("Waiting for location... Make sure to allow permissions in browser.")
else:
    user_lat, user_lon = None, None

# ========== GRMS ==========
threshold_table = pd.DataFrame()
grms_df = pd.DataFrame()
grms_enabled = st.checkbox("Enable GRMS Data")

if grms_enabled:
    grms_files = sorted([f for f in os.listdir("tecfiles") if f.endswith(".csv")])
    selected_file = st.selectbox("Choose GRMS File", grms_files)
    grms_df = load_grms_file(os.path.join("tecfiles", selected_file))

    if not grms_df.empty and "MP" in grms_df.columns:
        available_channels = [
            col for col in grms_df.columns
            if grms_df[col].dtype in [np.float64, np.int64]
        ]
        default_channel = next((col for col in available_channels if "Gage" in col), available_channels[0])
        selected_channels = st.multiselect("Choose up to 3 channels", available_channels, default=[default_channel])
        selected_channels = selected_channels[:3]

        center_mp = None
        if user_lat and user_lon:
            coords = list(zip(grms_df['Latitude'], grms_df['Longitude']))
            tree = KDTree(coords)
            dist, idx = tree.query([user_lat, user_lon])
            center_mp = grms_df.iloc[idx]['MP']

        auto_scroll = st.checkbox("Auto-scroll to my MP", value=False)
        recenter_btn = st.button("Recenter")

        fig = go.Figure()
        for col in selected_channels:
            fig.add_trace(go.Scatter(x=grms_df['MP'], y=grms_df[col], mode='lines', name=col))
        fig.update_layout(title="GRMS Channels", xaxis_title="MP", yaxis_title="Value", height=400)

        if (auto_scroll or recenter_btn) and center_mp:
            fig.update_xaxes(range=[center_mp - 0.05, center_mp + 0.05])

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Threshold Filters")
        for col in selected_channels:
            col_abs = st.checkbox(f"Use absolute value for {col}?", value=True)
            greater_than = st.number_input(f"{col} > ", value=1e10)
            less_than = st.number_input(f"{col} < ", value=-1e10)
            col_data = grms_df[col].abs() if col_abs else grms_df[col]
            match = grms_df[(col_data > greater_than) | (col_data < less_than)].copy()
            match['Flagged Channel'] = col
            threshold_table = pd.concat([threshold_table, match])

# ========== MAP ==========
st.subheader("🗺️ Geographic Map")

map_center = [40.7128, -74.0060]
zoom = 5

if user_lat and user_lon:
    map_center = [user_lat, user_lon]
    zoom = 12
elif not dtn_df.empty:
    map_center = [dtn_df['Latitude'].mean(), dtn_df['Longitude'].mean()]
    zoom = 8

m = folium.Map(location=map_center, zoom_start=zoom)

for _, row in dtn_df.iterrows():
    folium.CircleMarker(location=(row['Latitude'], row['Longitude']), radius=5, color='green').add_to(m)

for _, row in tec_df.iterrows():
    folium.CircleMarker(location=(row['Latitude'], row['Longitude']), radius=5, color='blue').add_to(m)

if not threshold_table.empty:
    for _, row in threshold_table.iterrows():
        folium.Marker(location=(row['Latitude'], row['Longitude']),
                      icon=folium.Icon(color="red")).add_to(m)

# Add Live Location
if user_lat and user_lon:
    folium.CircleMarker(
        location=[user_lat, user_lon],
        radius=8,
        color='#FF0000',
        fill=True,
        fill_color='#FF0000',
        popup="📍 Your Live Location",
        z_index_offset=1000
    ).add_to(m)

st_folium(m, width=1200, key="main_map")

# ========== TABLES ==========
st.subheader("Filtered DTN Data")
st.dataframe(dtn_df)

st.subheader("Filtered TEC Data")
st.dataframe(tec_df)
