import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from scipy.spatial import KDTree
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation  

# ========== CONFIG ==========
st.set_page_config(layout="wide")
st.title("Track Condition Dashboard")

# ========== VERTICAL POPUP ==========
def vertical_popup(row):
    html = "<table style='font-size:12px;'>"
    for key, value in row.items():
        html += f"<tr><th align='left'>{key}</th><td>{value}</td></tr>"
    html += "</table>"
    return folium.Popup(html, max_width=400)

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
            st.error(f"Missing columns in {os.path.basename(filepath)}: {', '.join(missing)}")
            return pd.DataFrame()
        df["# Milepost"] = pd.to_numeric(df["# Milepost"], errors="coerce")
        df["Milepost Feet"] = pd.to_numeric(df["Milepost Feet"], errors="coerce")
        df = df.dropna(subset=["# Milepost", "Milepost Feet", "Latitude", "Longitude"])
        df["MP"] = df["# Milepost"] + df["Milepost Feet"] / 5280
        if len(df) > 100000:
            df = df.iloc[::5]
        return df
    except Exception as e:
        st.error(f"Error reading GRMS file: {e}")
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

# ========== SESSION STATE ==========
if 'location_allowed' not in st.session_state:
    st.session_state['location_allowed'] = False
if 'user_lat' not in st.session_state:
    st.session_state['user_lat'] = None
if 'user_lon' not in st.session_state:
    st.session_state['user_lon'] = None

# ========== LOCATION TRACKING w/ REFRESH ==========
st.subheader("Live Location")
# put the checkbox and button side by side
col1, col2 = st.columns([4,1])
with col1:
    allow_live = st.checkbox(
        "Allow live location tracking",
        value=st.session_state['location_allowed'],
        help="Enable this to fetch your live location from your browser."
    )
with col2:
    refresh_clicked = st.button("🔄")

# Decide whether we need to call the geolocation component
if allow_live or refresh_clicked:
    location = streamlit_geolocation()
    if location and location.get("latitude") and location.get("longitude"):
        st.session_state['user_lat'] = location["latitude"]
        st.session_state['user_lon'] = location["longitude"]
        st.session_state['location_allowed'] = True
        if refresh_clicked:
            st.success("🔄 Location refreshed!")
        else:
            st.success("📍 Location captured!")
    else:
        st.info("📡 Waiting for browser permission or unable to fetch location.")
else:
    # if user turned off live-tracking, clear out
    st.session_state['location_allowed'] = False
    st.session_state['user_lat'] = None
    st.session_state['user_lon'] = None

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
        if available_channels:
            default_channel = next((c for c in available_channels if "Gage" in c), available_channels[0])
            selected_channels = st.multiselect("Choose up to 3 channels", available_channels, default=[default_channel])[:3]

            center_mp = None
            if all(k in grms_df.columns for k in ("Latitude","Longitude","MP")) and \
               st.session_state['user_lat'] and st.session_state['user_lon']:
                coords = list(zip(grms_df['Latitude'], grms_df['Longitude']))
                idx = KDTree(coords).query([st.session_state['user_lat'], st.session_state['user_lon']])[1]
                center_mp = grms_df.iloc[idx]['MP']

            auto_scroll = st.checkbox("Auto‑scroll Line Graph", value=False)
            recenter_btn = st.button("Recenter Graph")

            fig = go.Figure()
            for col in selected_channels:
                fig.add_trace(go.Scatter(x=grms_df['MP'], y=grms_df[col], mode='lines', name=col))
            fig.update_layout(title="GRMS Channels", xaxis_title="MP", yaxis_title="Value", height=400)
            if (auto_scroll or recenter_btn) and center_mp is not None:
                fig.update_xaxes(range=[center_mp-0.05, center_mp+0.05])
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Threshold Filters")
            for col in selected_channels:
                use_abs = st.checkbox(f"Use absolute for {col}?", value=True)
                gt = st.number_input(f"{col} >", value=1e10)
                lt = st.number_input(f"{col} <", value=-1e10)
                data_series = grms_df[col].abs() if use_abs else grms_df[col]
                flagged = grms_df[(data_series>gt)|(data_series<lt)].copy()
                flagged['Flagged Channel'] = col
                threshold_table = pd.concat([threshold_table, flagged])

# ========== MAP ==========
st.subheader("Geographic Map")
m = folium.Map(location=[40.7128, -74.0060], zoom_start=5)

# center on user if available
if st.session_state['user_lat'] and st.session_state['user_lon']:
    m.location = [st.session_state['user_lat'], st.session_state['user_lon']]
    m.zoom_start = 12
elif not dtn_df.empty:
    m.location = [dtn_df['Latitude'].mean(), dtn_df['Longitude'].mean()]
    m.zoom_start = 8

# DTN, TEC, GRMS and user‐marker
for _, r in dtn_df.iterrows():
    folium.CircleMarker((r['Latitude'], r['Longitude']), radius=5, color='green',
                        popup=vertical_popup(r)).add_to(m)
for _, r in tec_df.iterrows():
    folium.CircleMarker((r['Latitude'], r['Longitude']), radius=5, color='blue',
                        popup=vertical_popup(r)).add_to(m)
if not threshold_table.empty:
    for _, r in threshold_table.iterrows():
        folium.Marker((r['Latitude'], r['Longitude']),
                      popup=vertical_popup(r),
                      icon=folium.Icon(color="red")).add_to(m)
if st.session_state['user_lat'] and st.session_state['user_lon']:
    folium.CircleMarker([st.session_state['user_lat'], st.session_state['user_lon']],
                        radius=8, color='#F00', fill=True, fill_color='#F00',
                        popup="You", z_index_offset=1000).add_to(m)

st_folium(m, width=1200, key='main_map')

# ========== TABLES ==========
st.subheader("Filtered DTN Data")
st.dataframe(dtn_df)
st.subheader("Filtered TEC Data")
st.dataframe(tec_df)
