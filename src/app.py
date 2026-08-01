import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Configure relative imports dynamically for GitHub CI/CD compatibility
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai_doctor import CropDoctorAI
from satellite_engine import SatelliteProcessor

# 1. Page Configuration (Must be first Streamlit command)
st.set_page_config(page_title="Satellite AI Crop Doctor", layout="wide")

# 2. Header Section
st.title("🛰️ Satellite AI-Powered Crop Doctor")
st.markdown("Monitor crop health and detect drought or disease stress via satellite telemetry.")

# 3. Sidebar Configuration
st.sidebar.header("📍 Field Parameters")
crop_type = st.sidebar.selectbox(
    "Select Crop Type", 
    ["Wheat", "Maize / Corn", "Rice", "Cotton", "Soybean"]
)
latitude = st.sidebar.number_input("Field Latitude", value=16.5062, format="%.4f")
longitude = st.sidebar.number_input("Field Longitude", value=80.6480, format="%.4f")

# 4. Engine Initialization
processor = SatelliteProcessor()
ai_doctor = CropDoctorAI()

# Synthetic Band Data Generation
np.random.seed(42)
grid_size = (100, 100)

red_band = np.random.uniform(0.05, 0.25, grid_size)
nir_band = np.random.uniform(0.20, 0.60, grid_size)
swir_band = np.random.uniform(0.10, 0.35, grid_size)

# Simulate disease spot in field grid
nir_band[30:60, 30:60] -= 0.15 
red_band[30:60, 30:60] += 0.08

# 5. Execute Analysis
results = processor.analyze_field(red_band, nir_band, swir_band)
diagnosis = ai_doctor.diagnose(results, crop_type=crop_type)

# 6. Display High-Level Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean NDVI (Biomass)", f"{float(results['mean_ndvi']):.2f}")
col2.metric("Mean NDWI (Moisture)", f"{float(results['mean_ndwi']):.2f}")
col3.metric("Health Status", str(results['status']))
col4.metric("Stress Level", str(results['stress_level']))

st.markdown("---")

# 7. Maps & Prescriptions Layout
col_map, col_report = st.columns([1, 1])

with col_map:
    st.subheader("🗺️ Satellite Index Heatmap")
    layer_option = st.selectbox(
        "Select Map Layer", 
        ["NDVI Heatmap (Greenness)", "NDWI Map (Moisture)"]
    )
    
    fig, ax = plt.subplots(figsize=(6, 5))
    if "NDVI" in layer_option:
        im = ax.imshow(results['ndvi_map'], cmap='YlGn')
        plt.colorbar(im, ax=ax, label="NDVI Index")
        ax.set_title("Vegetation Vigour (NDVI)")
    else:
        im = ax.imshow(results['ndwi_map'], cmap='Blues')
        plt.colorbar(im, ax=ax, label="NDWI Index")
        ax.set_title("Canopy Water Content (NDWI)")
        
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

with col_report:
    st.subheader("🩺 AI Doctor Prescription")
    st.write(f"**Target Crop:** {diagnosis['crop']}")
    st.write(f"**Condition:** {diagnosis['condition']}")
    
    st.markdown("#### 🚨 Detected Issues:")
    for cause in diagnosis["probable_causes"]:
        st.write(f"- {cause}")

    st.markdown("#### 💡 Action Plan:")
    for rx in diagnosis["treatments"]:
        st.write(f"- {rx}")
        
    st.markdown("#### 💧 Irrigation Guidance:")
    st.info(diagnosis["irrigation_advice"])