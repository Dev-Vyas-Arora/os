import streamlit as st
import pandas as pd
import requests
import random
import joblib
from sklearn.cluster import KMeans
from fpdf import FPDF
import io
import numpy as np
import math

# --------------------------
# Load Saved ML Model and Encoders
# --------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("delivery_priority_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_type = joblib.load("delivery_type_encoder.pkl")
    le_priority = joblib.load("priority_encoder.pkl")
    return model, scaler, le_type, le_priority

# --------------------------
# Define Centres
# --------------------------
centres = {
    "Clement Town": {"lat": 30.3640, "lon": 78.0480},
    "Karanpur": {"lat": 30.3226, "lon": 78.0433},
    "Rajpur Road": {"lat": 30.3410, "lon": 78.0590}
}

def fallback_distances(centre_coord, df):
    dist_km = []
    time_min = []
    for _, row in df.iterrows():
        lat1, lon1 = centre_coord['lat'], centre_coord['lon']
        lat2, lon2 = row['lat'], row['long']
        km_lat = (lat2 - lat1) * 111
        km_lon = (lon2 - lon1) * 111 * math.cos(math.radians(lat1))
        distance = math.sqrt(km_lat**2 + km_lon**2)
        dist_km.append(round(distance, 2))
        time_min.append(round(distance * 2, 1))
    return dist_km, time_min

# --------------------------
# Fast Distance via OSRM Table API
# --------------------------
def get_osrm_table_distances(centre_coord, df):
    base = f"{centre_coord['lon']},{centre_coord['lat']}"
    coords = ";".join([f"{lon},{lat}" for lat, lon in zip(df["lat"], df["long"])])
    url = f"http://router.project-osrm.org/table/v1/driving/{base};{coords}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            dist_km = [round(d / 1000, 2) for d in data["distances"][0][1:]]
            time_min = [round(t / 60, 1) for t in data["durations"][0][1:]]
            return dist_km, time_min
    except:
        pass
    return fallback_distances(centre_coord, df)

# --------------------------
# Cluster Packages by Location
# --------------------------
def cluster_packages(packages_df, n_clusters):
    coords = packages_df[["lat", "long"]].values
    if len(coords) <= 1:
        packages_df["cluster"] = 0
        return packages_df
    k = min(n_clusters, len(coords))
    kmeans = KMeans(n_clusters=k, random_state=42)
    packages_df["cluster"] = kmeans.fit_predict(coords)
    return packages_df

# --------------------------
# Hybrid Scheduler
# --------------------------
def hybrid_scheduler_cluster_strict(df, drivers, capacities):
    driver_schedules = {driver: [] for driver in drivers}
    driver_index = 0
    priority_order = {"High": 0, "Medium": 1, "Low": 2}

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_pkgs = df[df["cluster"] == cluster_id]
        cluster_pkgs = cluster_pkgs.sort_values(
            by=["Priority", "Deadline_Hours"],
            key=lambda col: col.map(priority_order).fillna(3)
        )
        for _, pkg in cluster_pkgs.iterrows():
            driver = drivers[driver_index]
            if len(driver_schedules[driver]) >= capacities[driver]:
                driver_index = (driver_index + 1) % len(drivers)
                driver = drivers[driver_index]
            driver_schedules[driver].append(pkg.to_dict())
            driver_index = (driver_index + 1) % len(drivers)
    return driver_schedules

# --------------------------
# PDF Generator
# --------------------------
def create_pdf(schedules):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Driver Delivery Schedules", ln=True, align='C')
    pdf.ln(5)
    for driver, pkgs in schedules.items():
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 8, f"{driver} Schedule", ln=True)
        if pkgs:
            pdf.set_font("Arial", '', 12)
            for i, pkg in enumerate(pkgs, 1):
                pdf.multi_cell(
                    0, 6,
                    f"{i}. ID:{pkg['Product_ID']} | Type:{pkg['Delivery_Type']} | Priority:{pkg['Priority']} "
                    f"| Distance:{pkg['Distance (km)']} km | Deadline:{pkg['Deadline_Hours']} hrs"
                )
        else:
            pdf.cell(0, 6, "No packages assigned", ln=True)
        pdf.ln(4)
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return io.BytesIO(pdf_bytes)

# --------------------------
# Streamlit UI
# --------------------------
st.title("⚡ Driver Scheduler – Fast Version")

uploaded_file = st.file_uploader("📂 Upload delivery Data xlsx file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("📦 Uploaded Data Preview:")
    st.data_editor(df.head(10), disabled=True)

    # Select centre
    centre_choice = st.selectbox("Select Delivery Centre", list(centres.keys()))
    centre_coord = centres[centre_choice]

    # Compute distances
    st.info("🔄 Calculating distances using OSRM Table API...")
    df["Distance (km)"], df["Time (min)"] = get_osrm_table_distances(centre_coord, df)

    # Load model and encoders
    model, scaler, le_type, le_priority = load_assets()

    # Encode categorical columns
    df["Delivery_Type"] = le_type.transform(df["Delivery_Type"])

    # Prepare features
    features = df[["Weight (kg)", "Distance (km)", "Delivery_Type", "Value (₹)", "Fragile", "Deadline_Hours"]]
    features_scaled = scaler.transform(features)

    # Predict priorities
    pred_encoded = model.predict(features_scaled)
    df["Priority"] = le_priority.inverse_transform(pred_encoded)

    st.success("✅ Priorities Predicted Successfully!")
    st.data_editor(df.head(10), disabled=True)

    # Driver inputs
    num_drivers = st.number_input("Number of Drivers", min_value=1, value=2)
    drivers, driver_capacity = [], {}
    for i in range(num_drivers):
        name = st.text_input(f"Driver {i+1} Name", f"Driver{i+1}")
        cap = st.number_input(f"Max Packages for {name}", min_value=1, value=5)
        drivers.append(name)
        driver_capacity[name] = cap

    # Generate schedules
    if st.button("Generate Schedules"):
        df_clustered = cluster_packages(df, num_drivers * 2)
        schedules = hybrid_scheduler_cluster_strict(df_clustered, drivers, driver_capacity)
        st.session_state["schedules"] = schedules
        st.session_state["centre_coord"] = centre_coord
        st.session_state["centre_choice"] = centre_choice
        st.success("✅ Schedules Generated Successfully!")

# ✅ Keep dropdown & maps link persistent
if "schedules" in st.session_state:
    schedules = st.session_state["schedules"]
    centre_coord = st.session_state["centre_coord"]
    centre_choice = st.session_state["centre_choice"]

    driver_choice = st.selectbox("🚚 Choose Driver to View Google Maps Route", list(schedules.keys()))

    if driver_choice and schedules[driver_choice]:
        coordinates = [(centre_coord["lat"], centre_coord["lon"])]  # Start at centre
        for pkg in schedules[driver_choice]:
            coordinates.append((pkg["lat"], pkg["long"]))
        coordinates.append((centre_coord["lat"], centre_coord["lon"]))  # Return to centre

        start_lat, start_lng = coordinates[0]
        end_lat, end_lng = coordinates[-1]
        via_coords = coordinates[1:-1][::max(len(coordinates) // 5, 1)]
        waypoints = "/".join([f"{lat},{lng}" for lat, lng in via_coords])
        maps_url = f"https://www.google.com/maps/dir/{start_lat},{start_lng}/{waypoints}/{end_lat},{end_lng}"

        st.markdown(f"### 🗺️ [Open Route in Google Maps]({maps_url})")

    # Show schedules in tables
    for d, pkgs in schedules.items():
        st.subheader(f"📋 {d}'s Schedule")
        if pkgs:
            st.table(pd.DataFrame(pkgs)[["Product_ID", "Delivery_Type", "Priority", "Distance (km)", "Deadline_Hours"]])
        else:
            st.write("No packages assigned")

    # PDF Download
    pdf_buf = create_pdf(schedules)
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_buf,
        file_name="driver_schedules.pdf",
        mime="application/pdf"
    )
