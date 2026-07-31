import numpy as np

class SatelliteProcessor:
    """
    Processes multispectral satellite bands (e.g., Sentinel-2 B4, B8, B11)
    to generate vegetation health indices.
    """
    
    @staticmethod
    def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Normalized Difference Vegetation Index (NDVI) - Measures crop biomass."""
        denominator = nir + red
        denominator[denominator == 0] = 1e-6  # Prevent division by zero
        return (nir - red) / denominator

    @staticmethod
    def calculate_ndwi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """Normalized Difference Water Index (NDWI) - Measures crop moisture content."""
        denominator = nir + swir
        denominator[denominator == 0] = 1e-6
        return (nir - swir) / denominator

    def analyze_field(self, red_band: np.ndarray, nir_band: np.ndarray, swir_band: np.ndarray) -> dict:
        """Calculates field statistics and classifies crop health status."""
        ndvi = self.calculate_ndvi(red_band, nir_band)
        ndwi = self.calculate_ndwi(nir_band, swir_band)

        mean_ndvi = float(np.mean(ndvi))
        mean_ndwi = float(np.mean(ndwi))

        if mean_ndvi > 0.6 and mean_ndwi > 0.2:
            status = "Optimal Health"
            stress_level = "Low"
        elif mean_ndvi > 0.4 and mean_ndwi <= 0.1:
            status = "Water / Drought Stress"
            stress_level = "Moderate"
        elif mean_ndvi <= 0.3:
            status = "Severe Biomass Loss / Disease Risk"
            stress_level = "High"
        else:
            status = "Moderate Crop Stress"
            stress_level = "Medium"

        return {
            "mean_ndvi": round(mean_ndvi, 3),
            "mean_ndwi": round(mean_ndwi, 3),
            "status": status,
            "stress_level": stress_level,
            "ndvi_map": ndvi,
            "ndwi_map": ndwi
        }