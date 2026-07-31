import sys
import os

# Add parent directory (root folder) to Python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now all your `from src.xxx import yyy` lines will work without throwing errors:
from src.ai_doctor import CropDoctorAI
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from src.satellite_engine import SatelliteProcessor
from src.ai_doctor import CropDoctorAI

st.set_page_config(page_title="Satellite AI Crop Doctor", layout="wide")

st.title("🛰️ Satellite AI-Powered Crop Doctor")
st.markdown("Monitor crop health and detect drought or disease stress via satellite telemetry.")

# Sidebar Configuration
st.sidebar.header("📍 Field Parameters")
crop_type = st.sidebar.selectbox("Select Crop Type", ["Wheat", "Maize / Corn", "Rice", "Cotton", "Soybean"])
latitude = st.sidebar.number_input("Field Latitude", value=16.5062, format="%.4f")
longitude = st.sidebar.number_input("Field Longitude", value=80.6480, format="%.4f")

# Initialize Engine
processor = SatelliteProcessor()
ai_doctor = CropDoctorAI()

# Synthetic Band Data Generation for Interactive Testing
np.random.seed(42)
grid_size = (100, 100)

red_band = np.random.uniform(0.05, 0.25, grid_size)
nir_band = np.random.uniform(0.20, 0.60, grid_size)
swir_band = np.random.uniform(0.10, 0.35, grid_size)

# Simulate a disease spot in the middle of the field grid
nir_band[30:60, 30:60] -= 0.15 
red_band[30:60, 30:60] += 0.08

# Run Satellite Analysis
results = processor.analyze_field(red_band, nir_band, swir_band)
diagnosis = ai_doctor.diagnose(results, crop_type=crop_type)

# Display High-Level Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean NDVI (Biomass)", f"{results['mean_ndvi']}")
col2.metric("Mean NDWI (Moisture)", f"{results['mean_ndwi']}")
col3.metric("Health Status", results['status'])
col4.metric("Stress Level", results['stress_level'])

st.markdown("---")

col_map, col_report = st.columns([1, 1])

with col_map:
    st.subheader("🗺️ Satellite Index Heatmap")
    layer_option = st.selectbox("Select Map Layer", ["NDVI Heatmap (Greenness)", "NDWI Map (Moisture)"])
    
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