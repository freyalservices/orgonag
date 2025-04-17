import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from scipy.spatial import KDTree
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")
st.title("Track Condition Dashboard")

# ---------------- HELPERS ----------------
def vertical_popup(row: pd.Series) -> folium.Popup:
    html = "<table style='font-size:12px;'>"
    for key, value in row.items():
        html += f"<tr><th align='left'>{key}</th><td>{value}</td></tr>"
    html += "</table>"
    return folium.Popup(html, max_width=400)

@st.cache_data
def load_dtn(sub):
    df = pd.read_csv("dataset/dtn.csv")
    return df[df["Subdivision"] == sub]

@st.cache_data
def load_tec(sub):
    df = pd.read_csv("dataset/tec.csv")
    return df[df["Subdivision"] == sub]

@st.cache_data
def load_grms_file(path):
    try:
        df = pd.concat(pd.read_csv(path, chunksize=50_000))
        df.columns = [c.strip() for c in df.columns]
        req = ["# Milepost", "Milepost Feet", "Latitude", "Longitude"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            st.error(f"Missing columns in {os.path.basename(path)}: {', '.join(missing)}")
            return pd.DataFrame()
        df["# Milepost"] = pd.to_numeric(df["# Milepost"], errors="coerce")
        df["Milepost Feet"] = pd.to_numeric(df["Milepost Feet"], errors="coerce")
        df = df.dropna(subset=req)
        df["MP"] = df["# Milepost"] + df["Milepost Feet"] / 5280
        if len(df) > 100_000:
            df = df.iloc[::5]         # thin large file for speed
        return df
    except Exception as e:
        st.error(f"GRMS load error: {e}")
        return pd.DataFrame()

# ---------------- GLOBAL FILTERS ----------------
st.subheader("Global Filters")
dtn_raw = pd.read_csv("dataset/dtn.csv")
subdivision = st.selectbox("Choose Subdivision", dtn_raw["Subdivision"].dropna().unique())
date_range = st.date_input("Optional: Date Range", [])

# ---------------- DATA LOAD ----------------
dtn_df = load_dtn(subdivision)
tec_df = load_tec(subdivision)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Refine Filters")
status_filter  = st.sidebar.multiselect("DTN Status",  dtn_df["Status"].dropna().unique())
severity_filter= st.sidebar.multiselect("TEC Severity",tec_df["Severity"].dropna().unique())
user_review_filter = st.sidebar.multiselect("User Review Status",
                                           tec_df["User Review Status"].dropna().unique())

if status_filter:
    dtn_df = dtn_df[dtn_df["Status"].isin(status_filter)]
if severity_filter:
    tec_df = tec_df[tec_df["Severity"].isin(severity_filter)]
if user_review_filter:
    tec_df = tec_df[tec_df["User Review Status"].isin(user_review_filter)]

# ---------------- SESSION DEFAULTS ----------------
for k, v in {"location_allowed": False,
             "user_lat": None,
             "user_lon": None,
             "geo_counter": 0}.items():
    st.session_state.setdefault(k, v)

# ---------------- LOCATION TRACKING ----------------
allow_live_tracking = st.checkbox(
    "Allow live location tracking",
    value=st.session_state["location_allowed"],
    help="Enable this to fetch your live location from your browser."
)

if allow_live_tracking and st.button("🔄 Refresh Location"):
    st.session_state["geo_counter"] += 1    # force a new key on each click

location = None
if allow_live_tracking:
    # each run uses same key unless user clicked refresh
    geo_key = f"geo_{st.session_state['geo_counter']}"
    location = streamlit_geolocation(key=geo_key)

if allow_live_tracking and location and location.get("latitude") and location.get("longitude"):
    st.session_state["user_lat"] = location["latitude"]
    st.session_state["user_lon"] = location["longitude"]
    st.session_state["location_allowed"] = True
    st.success(f"📍  Location updated! ({location['latitude']:.5f}, "
               f"{location['longitude']:.5f})")
elif allow_live_tracking and not (st.session_state["user_lat"] and st.session_state["user_lon"]):
    st.info("📡 Waiting for browser location permission...")
elif not allow_live_tracking:
    st.session_state.update({"location_allowed": False, "user_lat": None, "user_lon": None})

# ---------------- GRMS SECTION ----------------
threshold_table = pd.DataFrame()
grms_df = pd.DataFrame()

if st.checkbox("Enable GRMS Data"):
    grms_files = sorted(f for f in os.listdir("tecfiles") if f.endswith(".csv"))
    if grms_files:
        selected_file = st.selectbox("Choose GRMS File", grms_files)
        grms_df = load_grms_file(os.path.join("tecfiles", selected_file))

        if not grms_df.empty and "MP" in grms_df.columns:
            num_cols = [c for c in grms_df.columns if grms_df[c].dtype in (np.float64, np.int64)]
            default_channel = next((c for c in num_cols if "Gage" in c), num_cols[0] if num_cols else None)
            chosen = st.multiselect("Choose up to 3 channels", num_cols, default=[default_channel] if default_channel else [])
            chosen = chosen[:3]

            # find closest MP to user
            center_mp = None
            if st.session_state["user_lat"] and st.session_state["user_lon"]:
                tree = KDTree(list(zip(grms_df["Latitude"], grms_df["Longitude"])))
                _, idx = tree.query([st.session_state["user_lat"], st.session_state["user_lon"]])
                center_mp = grms_df.iloc[idx]["MP"]

            auto_scroll = st.checkbox("Auto‑scroll line graph to my location")
            recenter_btn = st.button("Recenter Graph Around Me")

            fig = go.Figure()
            for col in chosen:
                fig.add_trace(go.Scatter(x=grms_df["MP"], y=grms_df[col],
                                         mode="lines", name=col))
            if (auto_scroll or recenter_btn) and center_mp is not None:
                fig.update_xaxes(range=[center_mp - 0.05, center_mp + 0.05])

            fig.update_layout(title="GRMS Channels",
                              xaxis_title="MP", yaxis_title="Value", height=400)
            st.plotly_chart(fig, use_container_width=True)

# ---------------- MAP ----------------
st.subheader("Geographic Map")

start_loc = ([st.session_state["user_lat"], st.session_state["user_lon"]]
             if st.session_state["user_lat"] and st.session_state["user_lon"]
             else [40.7128, -74.0060])
zoom_start = 12 if st.session_state["user_lat"] else 5
m = folium.Map(location=start_loc, zoom_start=zoom_start)

for _, r in dtn_df.iterrows():
    folium.CircleMarker((r["Latitude"], r["Longitude"]), radius=5,
                        color="green", popup=vertical_popup(r)).add_to(m)
for _, r in tec_df.iterrows():
    folium.CircleMarker((r["Latitude"], r["Longitude"]), radius=5,
                        color="blue", popup=vertical_popup(r)).add_to(m)
if st.session_state["user_lat"] and st.session_state["user_lon"]:
    folium.Marker([st.session_state["user_lat"], st.session_state["user_lon"]],
                  popup="Your Location", icon=folium.Icon(color="red")).add_to(m)

st_folium(m, width=1200, key="main_map")

# ---------------- TABLES ----------------
st.subheader("Filtered DTN Data")
st.dataframe(dtn_df)

st.subheader("Filtered TEC Data")
st.dataframe(tec_df)
