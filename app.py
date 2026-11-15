
from pathlib import Path
from datetime import timedelta
import html
import io

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import streamlit.components.v1 as components


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FireWatch | Satellite Fire Intelligence",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = BASE_FOLDER / "data"

FILES = {
    "MODIS": DATA_FOLDER / "MODIS_C6_1_South_Asia_7d.csv",
    "NOAA-20": DATA_FOLDER / "J1_VIIRS_C2_South_Asia_7d.csv",
    "NOAA-21": DATA_FOLDER / "J2_VIIRS_C2_South_Asia_7d.csv"
}


# ============================================================
# 2. PROFESSIONAL DASHBOARD THEME
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #091321;
    color: #F3F6FA;
}

[data-testid="stSidebar"] {
    background-color: #101D2E;
    border-right: 1px solid #26374D;
}

[data-testid="stHeader"] {
    background-color: #091321;
}

h1, h2, h3 {
    color: #F4F7FB !important;
}

p, label, span {
    color: #D5DFEA;
}

.hero {
    background: linear-gradient(120deg, #142A43, #1C344E);
    border: 1px solid #304962;
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 22px;
}

.hero h1 {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #B9C9D9;
    margin-bottom: 0;
}

.kpi {
    background: linear-gradient(145deg, #15283D, #102033);
    border: 1px solid #2A4159;
    padding: 19px;
    border-radius: 13px;
    min-height: 120px;
}

.kpi-label {
    color: #AFC0D2;
    font-size: 13px;
    font-weight: 600;
}

.kpi-value {
    color: #FFFFFF;
    font-size: 29px;
    font-weight: 800;
    margin-top: 10px;
}

.kpi-note {
    color: #F5A623;
    font-size: 12px;
    margin-top: 5px;
}

.section-heading {
    font-size: 21px;
    font-weight: 700;
    color: #F5F7FA;
    margin-top: 20px;
    margin-bottom: 12px;
}

.info-card {
    background: #13263B;
    border: 1px solid #2A4057;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 10px;
}

.small-muted {
    color: #A9B9CA;
    font-size: 12px;
}

.stButton button, .stDownloadButton button {
    background-color: #F28C28;
    color: #101D2E;
    border: none;
    border-radius: 8px;
    font-weight: 700;
}

.stButton button:hover, .stDownloadButton button:hover {
    background-color: #FFAD50;
    color: #101D2E;
}

hr {
    border-color: #2A4057;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def safe_number(value):
    return pd.to_numeric(value, errors="coerce")


def make_html_table(data, max_rows=15):
    if data is None or data.empty:
        st.info("No records available for this selection.")
        return

    display_data = data.head(max_rows).copy()

    for column in display_data.columns:
        display_data[column] = display_data[column].map(
            lambda x: html.escape(str(x))
        )

    table_html = display_data.to_html(
        index=False,
        border=0,
        classes="fire-table"
    )

    st.markdown("""
    <style>
    .fire-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        background: #13263B;
        color: #E8EFF6;
    }
    .fire-table th {
        background: #203A55;
        color: #FFFFFF;
        padding: 10px;
        text-align: left;
        border-bottom: 1px solid #405A73;
    }
    .fire-table td {
        padding: 9px;
        border-bottom: 1px solid #2A4057;
    }
    .fire-table tr:hover {
        background: #203A55;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        table_html,
        unsafe_allow_html=True
    )


def format_number(value):
    return f"{value:,.0f}"


def make_kpi(label, value, note):
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def create_chart_style(ax, title, xlabel="", ylabel=""):
    ax.set_title(
        title,
        color="white",
        fontsize=13,
        fontweight="bold",
        pad=15
    )
    ax.set_xlabel(xlabel, color="#C8D5E3")
    ax.set_ylabel(ylabel, color="#C8D5E3")
    ax.tick_params(colors="#C8D5E3")
    ax.set_facecolor("#13263B")
    ax.grid(
        True,
        alpha=0.18,
        linestyle="--"
    )


# ============================================================
# 4. COUNTRY CLASSIFICATION
# Approximate coordinate boxes, not official boundaries.
# ============================================================

COUNTRY_BOXES = {
    "Afghanistan": (29.0, 38.5, 60.5, 75.0),
    "Pakistan": (23.0, 37.5, 60.5, 77.5),
    "Nepal": (26.3, 30.5, 80.0, 88.5),
    "Bhutan": (26.5, 28.5, 88.5, 92.2),
    "Bangladesh": (20.5, 26.8, 88.0, 92.8),
    "Sri Lanka": (5.5, 10.2, 79.0, 82.2),
    "India": (6.0, 37.5, 68.0, 97.5)
}


def assign_country(latitude, longitude):
    for country in [
        "Afghanistan",
        "Pakistan",
        "Nepal",
        "Bhutan",
        "Bangladesh",
        "Sri Lanka",
        "India"
    ]:
        south, north, west, east = COUNTRY_BOXES[country]

        if (
            south <= latitude <= north
            and west <= longitude <= east
        ):
            return country

    return "Other / Unclassified"


# ============================================================
# 5. LOAD AND STANDARDIZE NASA DATA
# ============================================================

@st.cache_data(show_spinner="Loading NASA satellite observations...")
def load_nasa_data():

    all_data = []

    for satellite_name, file_path in FILES.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {file_path.name}"
            )

        df = pd.read_csv(file_path)

        df.columns = [
            str(column).strip().lower()
            for column in df.columns
        ]

        # Standardize brightness field
        if "brightness" not in df.columns:
            if "bright_ti4" in df.columns:
                df["brightness"] = df["bright_ti4"]
            elif "bright_ti5" in df.columns:
                df["brightness"] = df["bright_ti5"]
            else:
                df["brightness"] = np.nan

        required_columns = [
            "latitude",
            "longitude",
            "frp"
        ]

        for column in required_columns:
            if column not in df.columns:
                df[column] = np.nan

        df["latitude"] = safe_number(df["latitude"])
        df["longitude"] = safe_number(df["longitude"])
        df["frp"] = safe_number(df["frp"])
        df["brightness"] = safe_number(df["brightness"])

        if "confidence" not in df.columns:
            df["confidence"] = "Unknown"

        if "acq_date" not in df.columns:
            df["acq_date"] = ""

        if "acq_time" not in df.columns:
            df["acq_time"] = "0000"

        df["acq_time"] = (
            df["acq_time"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(4)
        )

        df["acquisition_datetime"] = pd.to_datetime(
            df["acq_date"].astype(str)
            + " "
            + df["acq_time"].str[:2]
            + ":"
            + df["acq_time"].str[2:4],
            errors="coerce"
        )

        df["Satellite Source"] = satellite_name

        if satellite_name == "MODIS":
            df["Dataset"] = "MODIS"
        else:
            df["Dataset"] = "VIIRS"

        df = df.dropna(
            subset=["latitude", "longitude"]
        )

        df = df[
            df["latitude"].between(-90, 90)
            & df["longitude"].between(-180, 180)
        ]

        df["Country"] = df.apply(
            lambda row: assign_country(
                row["latitude"],
                row["longitude"]
            ),
            axis=1
        )

        all_data.append(df)

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    return combined


# ============================================================
# 6. LOAD DATA
# ============================================================

try:
    raw_data = load_nasa_data()

except Exception as error:
    st.error("Unable to load the NASA datasets.")
    st.error(str(error))
    st.stop()


if raw_data.empty:
    st.error("The NASA datasets contain no usable observations.")
    st.stop()


latest_date = raw_data["acquisition_datetime"].max()
earliest_date = raw_data["acquisition_datetime"].min()


# ============================================================
# 7. SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("## 🔥 FireWatch")
st.sidebar.caption("Satellite Fire Intelligence Platform")
st.sidebar.markdown("---")

st.sidebar.markdown("### Geographic Filter")

available_countries = sorted(
    raw_data["Country"].dropna().unique().tolist()
)

country_options = ["All South Asia"] + available_countries

selected_country = st.sidebar.selectbox(
    "Select country",
    country_options
)

st.sidebar.markdown("### Satellite Filter")

satellite_options = [
    "All Satellites",
    "MODIS",
    "NOAA-20",
    "NOAA-21"
]

selected_satellite = st.sidebar.selectbox(
    "Satellite source",
    satellite_options
)

st.sidebar.markdown("### Observation Period")

date_options = [
    "All Downloaded Data",
    "Last 24 Hours",
    "Last 48 Hours",
    "Last 7 Days"
]

selected_period = st.sidebar.selectbox(
    "Select observation period",
    date_options
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Dataset Information")

st.sidebar.write(
    f"**Total raw observations:** {format_number(len(raw_data))}"
)

st.sidebar.write(
    f"**First observation:** {earliest_date}"
)

st.sidebar.write(
    f"**Latest observation:** {latest_date}"
)

st.sidebar.caption(
    "Data source: downloaded NASA FIRMS CSV snapshots. "
    "This dashboard is not a live satellite feed."
)


# ============================================================
# 8. APPLY FILTERS
# ============================================================

data = raw_data.copy()

if selected_country != "All South Asia":
    data = data[
        data["Country"] == selected_country
    ]

if selected_satellite == "MODIS":
    data = data[
        data["Satellite Source"] == "MODIS"
    ]

elif selected_satellite == "NOAA-20":
    data = data[
        data["Satellite Source"] == "NOAA-20"
    ]

elif selected_satellite == "NOAA-21":
    data = data[
        data["Satellite Source"] == "NOAA-21"
    ]


if selected_period != "All Downloaded Data":

    if selected_period == "Last 24 Hours":
        cutoff = latest_date - timedelta(hours=24)

    elif selected_period == "Last 48 Hours":
        cutoff = latest_date - timedelta(hours=48)

    else:
        cutoff = latest_date - timedelta(days=7)

    data = data[
        data["acquisition_datetime"] >= cutoff
    ]


if data.empty:
    st.warning("No observations match the selected filters.")
    st.stop()


# ============================================================
# 9. HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🔥 FIREWATCH</h1>
    <h3>Satellite-Based Forest Fire Monitoring & Geospatial Intelligence</h3>
    <p>
        Monitoring recent fire detections across South Asia using
        NASA MODIS, NOAA-20 and NOAA-21 satellite observations.
    </p>
</div>
""", unsafe_allow_html=True)


st.caption(
    f"Showing {len(data):,} filtered observations | "
    f"Observation window: {selected_period} | "
    f"Country: {selected_country}"
)


# ============================================================
# 10. KEY PERFORMANCE INDICATORS
# ============================================================

st.markdown(
    '<div class="section-heading">Platform Overview</div>',
    unsafe_allow_html=True
)

total_hotspots = len(data)

average_frp = data["frp"].mean()

if pd.isna(average_frp):
    average_frp = 0

high_frp_count = int(
    (data["frp"] >= 50).sum()
)

active_satellites = data[
    "Satellite Source"
].nunique()

col1, col2, col3, col4 = st.columns(4)

with col1:
    make_kpi(
        "TOTAL HOTSPOT OBSERVATIONS",
        format_number(total_hotspots),
        "Filtered NASA detections"
    )

with col2:
    make_kpi(
        "ACTIVE SATELLITE SOURCES",
        str(active_satellites),
        "MODIS / VIIRS platforms"
    )

with col3:
    make_kpi(
        "AVERAGE FIRE RADIATIVE POWER",
        f"{average_frp:.2f} MW",
        "Average observed FRP"
    )

with col4:
    make_kpi(
        "HIGH FRP DETECTIONS",
        format_number(high_frp_count),
        "Project threshold: FRP ≥ 50 MW"
    )


# ============================================================
# 11. MAP PREPARATION
# ============================================================

st.markdown(
    '<div class="section-heading">Geospatial Fire Monitoring</div>',
    unsafe_allow_html=True
)

map_data = data.dropna(
    subset=["latitude", "longitude"]
).copy()

map_center = [
    map_data["latitude"].mean(),
    map_data["longitude"].mean()
]

fire_map = folium.Map(
    location=map_center,
    zoom_start=5,
    tiles="OpenStreetMap"
)

heat_points = map_data[
    ["latitude", "longitude"]
].values.tolist()

if heat_points:
    HeatMap(
        heat_points,
        radius=13,
        blur=18,
        min_opacity=0.35
    ).add_to(fire_map)


# Add individual hotspot markers
marker_data = map_data

if len(marker_data) > 1500:
    marker_data = marker_data.sample(
        1500,
        random_state=42
    )

for _, row in marker_data.iterrows():

    frp_value = row["frp"]

    if pd.isna(frp_value):
        frp_value = 0

    if frp_value >= 50:
        marker_color = "red"
    elif frp_value >= 20:
        marker_color = "orange"
    else:
        marker_color = "green"

    popup_text = (
        f"Country: {row['Country']}<br>"
        f"Satellite: {row['Satellite Source']}<br>"
        f"FRP: {frp_value:.2f} MW<br>"
        f"Latitude: {row['latitude']:.4f}<br>"
        f"Longitude: {row['longitude']:.4f}"
    )

    folium.CircleMarker(
        location=[
            row["latitude"],
            row["longitude"]
        ],
        radius=4,
        color=marker_color,
        fill=True,
        fill_opacity=0.8,
        popup=folium.Popup(
            popup_text,
            max_width=250
        )
    ).add_to(fire_map)


components.html(
    fire_map.get_root().render(),
    height=570,
    scrolling=False
)

st.caption(
    "Heatmap intensity represents the concentration of recorded "
    "hotspot observations. Marker colors represent project-defined "
    "FRP categories, not official fire-danger ratings."
)


# ============================================================
# 12. DAILY FIRE ACTIVITY TREND
# ============================================================

st.markdown(
    '<div class="section-heading">Fire Activity Timeline</div>',
    unsafe_allow_html=True
)

trend_data = data.dropna(
    subset=["acquisition_datetime"]
).copy()

if not trend_data.empty:

    trend_data["Observation Date"] = (
        trend_data["acquisition_datetime"].dt.date
    )

    daily_counts = (
        trend_data.groupby("Observation Date")
        .size()
        .reset_index(name="Hotspots")
    )

    fig, ax = plt.subplots(figsize=(12, 4))

    fig.patch.set_facecolor("#091321")

    ax.plot(
        daily_counts["Observation Date"].astype(str),
        daily_counts["Hotspots"],
        marker="o",
        linewidth=2.5
    )

    ax.fill_between(
        range(len(daily_counts)),
        daily_counts["Hotspots"],
        alpha=0.15
    )

    create_chart_style(
        ax,
        "Daily Satellite Fire Observations",
        "Observation Date",
        "Number of detections"
    )

    ax.tick_params(axis="x", rotation=35)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# 13. SATELLITE COMPARISON
# ============================================================

st.markdown(
    '<div class="section-heading">Satellite Performance Comparison</div>',
    unsafe_allow_html=True
)

satellite_summary = (
    data.groupby("Satellite Source")
    .agg(
        Hotspots=("latitude", "count"),
        Average_FRP=("frp", "mean"),
        Maximum_FRP=("frp", "max")
    )
    .reset_index()
)

sat_col1, sat_col2 = st.columns(2)

with sat_col1:

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#091321")

    ax.bar(
        satellite_summary["Satellite Source"],
        satellite_summary["Hotspots"]
    )

    create_chart_style(
        ax,
        "Hotspot Observations by Satellite",
        "Satellite",
        "Detections"
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


with sat_col2:

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#091321")

    ax.bar(
        satellite_summary["Satellite Source"],
        satellite_summary["Average_FRP"]
    )

    create_chart_style(
        ax,
        "Average FRP by Satellite",
        "Satellite",
        "Average FRP (MW)"
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


make_html_table(
    satellite_summary.round(2),
    max_rows=10
)


# ============================================================
# 14. FRP INTENSITY ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-heading">Fire Radiative Power Analysis</div>',
    unsafe_allow_html=True
)

frp_data = data.dropna(
    subset=["frp"]
).copy()

if not frp_data.empty:

    frp_data["Intensity Category"] = pd.cut(
        frp_data["frp"],
        bins=[
            -np.inf,
            20,
            50,
            np.inf
        ],
        labels=[
            "Low",
            "Moderate",
            "High"
        ]
    )

    intensity_summary = (
        frp_data["Intensity Category"]
        .value_counts()
        .reindex(
            ["Low", "Moderate", "High"],
            fill_value=0
        )
        .reset_index()
    )

    intensity_summary.columns = [
        "Intensity",
        "Observations"
    ]

    frp_col1, frp_col2 = st.columns(2)

    with frp_col1:

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#091321")

        ax.bar(
            intensity_summary["Intensity"],
            intensity_summary["Observations"]
        )

        create_chart_style(
            ax,
            "FRP Intensity Distribution",
            "Intensity category",
            "Observations"
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with frp_col2:

        st.markdown("#### FRP Statistics")

        st.metric(
            "Average FRP",
            f"{frp_data['frp'].mean():.2f} MW"
        )

        st.metric(
            "Maximum FRP",
            f"{frp_data['frp'].max():.2f} MW"
        )

        st.metric(
            "Minimum FRP",
            f"{frp_data['frp'].min():.2f} MW"
        )

        st.metric(
            "High FRP observations",
            format_number(
                (frp_data["frp"] >= 50).sum()
            )
        )

    st.caption(
        "Intensity thresholds are defined for this project: "
        "Low < 20 MW, Moderate 20–50 MW, High > 50 MW. "
        "They are not official fire-danger classifications."
    )


# ============================================================
# 15. CONFIDENCE ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-heading">Detection Confidence Analysis</div>',
    unsafe_allow_html=True
)

confidence_data = data.copy()

confidence_data["Confidence Display"] = (
    confidence_data["confidence"]
    .astype(str)
    .str.strip()
    .str.title()
)

confidence_data.loc[
    confidence_data["Confidence Display"].isin(
        ["", "Nan", "None"]
    ),
    "Confidence Display"
] = "Unknown"

confidence_summary = (
    confidence_data.groupby(
        ["Satellite Source", "Confidence Display"]
    )
    .size()
    .reset_index(name="Observations")
)

confidence_col1, confidence_col2 = st.columns(2)

with confidence_col1:

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#091321")

    confidence_pivot = confidence_summary.pivot(
        index="Satellite Source",
        columns="Confidence Display",
        values="Observations"
    ).fillna(0)

    confidence_pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax
    )

    create_chart_style(
        ax,
        "Confidence Distribution by Satellite",
        "Satellite",
        "Observations"
    )

    ax.legend(
        facecolor="#13263B",
        labelcolor="white"
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


with confidence_col2:

    st.markdown("#### Confidence Records")

    make_html_table(
        confidence_summary,
        max_rows=15
    )

st.caption(
    "MODIS and VIIRS confidence fields can use different formats. "
    "Their labels should not be interpreted as directly equivalent "
    "sensor-quality scores."
)


# ============================================================
# 16. GEOGRAPHIC RISK-ZONE ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-heading">Geographic Risk-Zone Intelligence</div>',
    unsafe_allow_html=True
)

risk_data = data.dropna(
    subset=["latitude", "longitude"]
).copy()

risk_data["Latitude Grid"] = np.floor(
    risk_data["latitude"]
)

risk_data["Longitude Grid"] = np.floor(
    risk_data["longitude"]
)

risk_zones = (
    risk_data.groupby(
        ["Latitude Grid", "Longitude Grid"]
    )
    .size()
    .reset_index(name="Hotspot Count")
)

if not risk_zones.empty:

    low_limit = risk_zones["Hotspot Count"].quantile(0.40)
    high_limit = risk_zones["Hotspot Count"].quantile(0.75)

    def classify_risk(count):

        if count >= high_limit:
            return "High"

        elif count >= low_limit:
            return "Moderate"

        return "Low"

    risk_zones["Risk Zone"] = (
        risk_zones["Hotspot Count"]
        .apply(classify_risk)
    )

    risk_col1, risk_col2, risk_col3 = st.columns(3)

    with risk_col1:
        make_kpi(
            "HIGH DENSITY ZONES",
            format_number(
                (risk_zones["Risk Zone"] == "High").sum()
            ),
            "Highest observation-density group"
        )

    with risk_col2:
        make_kpi(
            "MODERATE DENSITY ZONES",
            format_number(
                (risk_zones["Risk Zone"] == "Moderate").sum()
            ),
            "Intermediate observation-density group"
        )

    with risk_col3:
        make_kpi(
            "LOW DENSITY ZONES",
            format_number(
                (risk_zones["Risk Zone"] == "Low").sum()
            ),
            "Lowest observation-density group"
        )

    st.markdown("#### Top 15 Hotspot-Dense Geographic Zones")

    top_zones = risk_zones.sort_values(
        "Hotspot Count",
        ascending=False
    ).head(15)

    top_zones = top_zones.rename(
        columns={
            "Latitude Grid": "Grid Latitude",
            "Longitude Grid": "Grid Longitude"
        }
    )

    make_html_table(
        top_zones.round(2),
        max_rows=15
    )

    st.caption(
        "Risk zones are based on observed hotspot density in "
        "one-degree latitude/longitude grid cells. They are "
        "indicators of recorded activity, not predictions of "
        "future fires, fire spread or official emergency risk."
    )


# ============================================================
# 17. MULTI-SATELLITE OBSERVATION ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-heading">Multi-Satellite Observation Analysis</div>',
    unsafe_allow_html=True
)

corroboration_data = data.dropna(
    subset=["latitude", "longitude"]
).copy()

corroboration_data["Lat Cell"] = (
    corroboration_data["latitude"].round(1)
)

corroboration_data["Lon Cell"] = (
    corroboration_data["longitude"].round(1)
)

corroboration = (
    corroboration_data.groupby(
        ["Lat Cell", "Lon Cell"]
    )
    .agg(
        Satellite_Count=(
            "Satellite Source",
            "nunique"
        ),
        Observation_Count=(
            "Satellite Source",
            "count"
        ),
        Sources=(
            "Satellite Source",
            lambda x: ", ".join(sorted(set(x)))
        )
    )
    .reset_index()
)

multi_satellite = corroboration[
    corroboration["Satellite_Count"] > 1
]

multi_col1, multi_col2 = st.columns(2)

with multi_col1:

    make_kpi(
        "MULTI-SATELLITE GRID CELLS",
        format_number(len(multi_satellite)),
        "Rounded 0.1-degree coordinate cells"
    )

with multi_col2:

    make_kpi(
        "OBSERVATIONS IN THESE CELLS",
        format_number(
            multi_satellite["Observation_Count"].sum()
            if not multi_satellite.empty
            else 0
        ),
        "Combined observations"
    )

st.markdown("#### Multi-Satellite Grid Records")

make_html_table(
    multi_satellite.sort_values(
        "Observation_Count",
        ascending=False
    ),
    max_rows=15
)

st.caption(
    "Nearby observations from different satellites do not prove "
    "that they detected the exact same fire. Acquisition times, "
    "sensor resolution and location uncertainty can differ."
)


# ============================================================
# 18. COUNTRY-WISE FIRE ACTIVITY
# ============================================================

st.markdown(
    '<div class="section-heading">Country-Wise Fire Activity</div>',
    unsafe_allow_html=True
)

country_summary = (
    data.groupby("Country")
    .agg(
        Hotspots=("latitude", "count"),
        Average_FRP=("frp", "mean"),
        Maximum_FRP=("frp", "max")
    )
    .reset_index()
    .sort_values(
        "Hotspots",
        ascending=False
    )
)

country_col1, country_col2 = st.columns(2)

with country_col1:

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#091321")

    ax.barh(
        country_summary["Country"],
        country_summary["Hotspots"]
    )

    ax.invert_yaxis()

    create_chart_style(
        ax,
        "Hotspot Observations by Country",
        "Observations",
        "Country"
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with country_col2:

    make_html_table(
        country_summary.round(2),
        max_rows=15
    )

st.caption(
    "Country labels use approximate coordinate bounding boxes "
    "for dashboard grouping. They are not official administrative "
    "boundary determinations."
)


# ============================================================
# 19. FIRE DETECTION RECORDS
# ============================================================

st.markdown(
    '<div class="section-heading">Fire Detection Records</div>',
    unsafe_allow_html=True
)

record_columns = [
    "acquisition_datetime",
    "Country",
    "Satellite Source",
    "latitude",
    "longitude",
    "brightness",
    "frp",
    "confidence"
]

record_columns = [
    column
    for column in record_columns
    if column in data.columns
]

records = data[record_columns].copy()

records = records.sort_values(
    "acquisition_datetime",
    ascending=False
)

records = records.rename(
    columns={
        "acquisition_datetime": "Observation Time",
        "Satellite Source": "Satellite",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "brightness": "Brightness",
        "frp": "FRP (MW)",
        "confidence": "Confidence"
    }
)

st.markdown("#### Latest 25 Observations")

make_html_table(
    records,
    max_rows=25
)


# ============================================================
# 20. DOWNLOAD FILTERED DATA
# ============================================================

st.markdown(
    '<div class="section-heading">Export Fire Observation Data</div>',
    unsafe_allow_html=True
)

csv_data = data.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇ Download Filtered Fire Records (CSV)",
    data=csv_data,
    file_name="FireWatch_Filtered_Observations.csv",
    mime="text/csv"
)


# ============================================================
# 21. PROJECT INFORMATION
# ============================================================

st.markdown("---")

st.markdown("""
<div class="info-card">
    <h3>About FireWatch</h3>
    <p>
        FireWatch is a satellite-based forest fire monitoring and
        geospatial analysis project built using NASA FIRMS
        observations from MODIS, NOAA-20 VIIRS and NOAA-21 VIIRS.
    </p>
    <p>
        The platform analyzes hotspot distribution, fire radiative
        power, detection confidence, satellite observations and
        geographic hotspot density across South Asia.
    </p>
    <p class="small-muted">
        Data source: NASA FIRMS downloaded CSV datasets.
        This application analyzes the supplied data snapshots;
        it does not provide live emergency alerts or predict
        future fire events.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;color:#8498AD;font-size:12px;'>"
    "FIREWATCH | Satellite Fire Intelligence | NASA FIRMS Data"
    "</p>",
    unsafe_allow_html=True
)
