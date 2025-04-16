import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from scipy.spatial import KDTree
import folium

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
    """Load large CSV in chunks, required columns, parse numeric mileposts."""
    try:
        chunks = pd.read_csv(filepath, chunksize=50_000)
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

        # Downsample for large files
        if len(df) > 100_000:
            df = df.iloc[::5]

        return df
    except Exception as e:
        st.error(f"Error loading GRMS file: {e}")
        return pd.DataFrame()

# ========== GLOBAL FILTERS ==========
st.subheader("Global Filters")
dtn_raw = pd.read_csv("dataset/dtn.csv")

subdivision = st.selectbox(
    "Choose Subdivision",
    options=dtn_raw['Subdivision'].dropna().unique()
)
date_range = st.date_input("Optional: Date Range", [])

# ========== LOAD DATA ==========
dtn_df = load_dtn(subdivision)
tec_df = load_tec(subdivision)

# ========== SIDEBAR FILTERS ==========
st.sidebar.title("Refine Filters")
status_filter = st.sidebar.multiselect(
    "DTN Status", 
    options=dtn_df['Status'].dropna().unique()
)
severity_filter = st.sidebar.multiselect(
    "TEC Severity", 
    options=tec_df['Severity'].dropna().unique()
)
user_review_filter = st.sidebar.multiselect(
    "User Review Status", 
    options=tec_df['User Review Status'].dropna().unique()
)

if status_filter:
    dtn_df = dtn_df[dtn_df['Status'].isin(status_filter)]
if severity_filter:
    tec_df = tec_df[tec_df['Severity'].isin(severity_filter)]
if user_review_filter:
    tec_df = tec_df[tec_df['User Review Status'].isin(user_review_filter)]

# ========== LIVE TRACKING ==========
st.subheader("📍 Live Location Tracking")

track_enabled = st.checkbox(
    "Enable Live Tracking",
    help="Uses browser geolocation to track your current location without a full page refresh."
)

# Optional: Attempt to get one-time location from Python side 
# (streamlit-geolocation or any other method). 
# We'll gracefully handle if it's unavailable.
try:
    from streamlit_geolocation import streamlit_geolocation
    location = streamlit_geolocation()
    user_lat = location.get("latitude") if location else None
    user_lon = location.get("longitude") if location else None
except Exception as e:
    st.info("Geolocation component not available.")
    user_lat, user_lon = None, None

if track_enabled and user_lat and user_lon:
    st.success(f"Your initial location: ({user_lat:.5f}, {user_lon:.5f})")
else:
    st.info("Location not available or tracking not enabled. Map will use fallback center.")

# ========== GRMS ==========
threshold_table = pd.DataFrame()
grms_df = pd.DataFrame()
grms_enabled = st.checkbox("Enable GRMS Data")

if grms_enabled:
    # Let user pick a file from "tecfiles" folder
    grms_files = sorted([f for f in os.listdir("tecfiles") if f.endswith(".csv")])
    selected_file = st.selectbox("Choose GRMS File", grms_files)
    grms_df = load_grms_file(os.path.join("tecfiles", selected_file))

    if not grms_df.empty and "MP" in grms_df.columns:
        # Let user pick numeric channels
        numeric_cols = [
            col for col in grms_df.columns 
            if grms_df[col].dtype in [np.float64, np.int64]
        ]
        default_channel = next((col for col in numeric_cols if "Gage" in col), None)
        default_selection = [default_channel] if default_channel else numeric_cols[:1]
        selected_channels = st.multiselect(
            "Choose up to 3 channels", 
            numeric_cols,
            default=default_selection
        )
        selected_channels = selected_channels[:3]  # limit to 3

        # Attempt to center graph around user location if known
        center_mp = None
        if user_lat and user_lon:
            coords = list(zip(grms_df['Latitude'], grms_df['Longitude']))
            tree = KDTree(coords)
            dist, idx = tree.query([user_lat, user_lon])
            center_mp = grms_df.iloc[idx]['MP']

        auto_scroll = st.checkbox("Auto-scroll to my MP", value=False)
        recenter_btn = st.button("Recenter")

        # Create Plotly chart
        fig = go.Figure()
        for col in selected_channels:
            fig.add_trace(
                go.Scatter(
                    x=grms_df['MP'],
                    y=grms_df[col],
                    mode='lines',
                    name=col
                )
            )
        fig.update_layout(
            title="GRMS Channels",
            xaxis_title="MP",
            yaxis_title="Value",
            height=400
        )

        # Optionally recenter
        if (auto_scroll or recenter_btn) and center_mp:
            fig.update_xaxes(range=[center_mp - 0.05, center_mp + 0.05])

        st.plotly_chart(fig, use_container_width=True)

        # Threshold filters
        st.markdown("### Threshold Filters")
        for col in selected_channels:
            col_abs = st.checkbox(f"Use absolute value for {col}?", value=True)
            greater_than = st.number_input(f"{col} > ", value=1e10)
            less_than = st.number_input(f"{col} < ", value=-1e10)

            col_data = grms_df[col].abs() if col_abs else grms_df[col]
            match = grms_df[
                (col_data > greater_than) | (col_data < less_than)
            ].copy()
            match['Flagged Channel'] = col
            threshold_table = pd.concat([threshold_table, match])

# ========== MAP ==========
st.subheader("🗺️ Geographic Map")

# Determine fallback map center and zoom
if user_lat and user_lon:
    map_center = [user_lat, user_lon]
    zoom = 12
elif not dtn_df.empty:
    map_center = [
        dtn_df['Latitude'].mean(),
        dtn_df['Longitude'].mean()
    ]
    zoom = 8
else:
    # Fallback: NYC area
    map_center = [40.7128, -74.0060]
    zoom = 5

# Create Folium map
m = folium.Map(location=map_center, zoom_start=zoom)

# Add static markers for DTN data
for _, row in dtn_df.iterrows():
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=5, 
        color='green',
        fill=True
    ).add_to(m)

# Add static markers for TEC data
for _, row in tec_df.iterrows():
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=5, 
        color='blue',
        fill=True
    ).add_to(m)

# Add flagged threshold markers
if not threshold_table.empty:
    for _, row in threshold_table.iterrows():
        folium.Marker(
            location=(row['Latitude'], row['Longitude']),
            icon=folium.Icon(color="red")
        ).add_to(m)

# Render map HTML
map_html = m.get_root().render()

# If live tracking is enabled, inject watchPosition JS into the rendered HTML.
# This script runs on the client side in the top-level DOM (not an iframe).
if track_enabled:
    live_location_js = f"""
    <script>
    if (navigator.geolocation) {{
        // Place a red circleMarker at the initial center
        var liveMarker = L.circleMarker([{map_center[0]}, {map_center[1]}], {{
            radius: 8,
            color: '#FF0000',
            fill: true,
            fillColor: '#FF0000',
            fillOpacity: 1
        }}).addTo({m.get_name()});

        // Watch for position changes
        navigator.geolocation.watchPosition(function(position) {{
            var newPos = [position.coords.latitude, position.coords.longitude];
            liveMarker.setLatLng(newPos);

            // If you want the map to follow the user, uncomment:
            // {m.get_name()}.setView(newPos, {m.get_name()}.getZoom());
        }}, 
        function(error) {{
            console.log("Error watching position:", error);
        }}, 
        {{
            enableHighAccuracy: true,
            timeout: 60000,
            maximumAge: 0
        }});
    }} else {{
       console.log("Geolocation is not supported by this browser.");
    }}
    </script>
    """
    map_html += live_location_js

# Display the final map (including any custom JS)
st.markdown(map_html, unsafe_allow_html=True)

# ========== TABLES ==========
st.subheader("Filtered DTN Data")
st.dataframe(dtn_df)

st.subheader("Filtered TEC Data")
st.dataframe(tec_df)
