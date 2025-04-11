import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.measurements import Measurements
from scipy.stats import skewnorm

class SyntheticDataGenerator:
    """
    Using Variational Autoencoder to generate synthetic data.
    """

    def __init__(self):
        self.max_energy = 9963.0
        self.measurements = Measurements().get_measurements()

    def generate_synthetic_data(self, plot=False):

        id_num = np.random.randint(1, 100)

        unique_ids = self.measurements["ID_File"].unique()
        id_selected = unique_ids[id_num]

        measurements = self.measurements.loc[self.measurements["ID_File"] == id_selected].reset_index(drop=True)

        # Randomly determine the number of peaks
        num_peaks = np.random.randint(1, 5)
        peaks = np.random.uniform(1, 2366.11, num_peaks)  # Randomly placed peaks

        # Define energy range and Gaussian width
        energy = np.arange(0, 2366.11, 0.29)

        measurements["Count"] = measurements["Count"] + 1
        measurements["LogCount"] = np.log10(measurements["Count"]).round(3)

        log_reg = np.polyfit(energy, measurements["LogCount"], 20)
        model = np.polyfit(energy, measurements["Count"], 10)
        counts = 10 ** (np.polyval(log_reg, energy))
        measurements["pred"] = counts

        # Add noise to the baseline
        noise_level = np.random.randint(1, 3)
        noise_start = np.random.randint(3, 6)
        noise = np.random.normal(noise_start, noise_level, size=len(counts))
        counts += noise

        # Add Gaussian peaks
        all_gaussians = np.zeros_like(energy)  # Aggregate all Gaussian contributions
        is_anomalous = np.zeros_like(energy, dtype=bool)  # Initialize anomaly labels
        for peak in peaks:
            # Base parameters
            peak_width = 0.01 + 0.001 * peak
            intensity = np.random.randint(1000, 60001)

            # Variations
            peak_width_variation = peak_width * (1 + np.random.uniform(-0.1, 0.1))  # Width variation
            intensity_variation = intensity * (1 + np.random.uniform(-0.05, 0.05))  # Intensity variation
            local_noise = np.random.normal(0, 0.01 * intensity_variation, size=len(energy))  # Local noise

            # Skewness
            skewness = np.random.uniform(-5, 5)
            gaussian = intensity_variation * skewnorm.pdf(energy, skewness, loc=peak, scale=peak_width_variation)

            # Add noise to the Gaussian
            # gaussian += local_noise

            # Aggregate Gaussian contributions
            # counts += gaussian
            # all_gaussians += gaussian



            # Label anomalies around the peak
            anomaly_threshold = peak_width * 5  # Define a range around the peak
            is_anomalous |= np.abs(energy - peak) <= anomaly_threshold

        # Add swing noise (cosine wave)
        swing_amplitude = np.random.randint(100, 200)
        swing_frequency = 4 / (2 * (max(energy) - min(energy)))
        # swings = swing_amplitude * np.cos(2 * np.pi * swing_frequency * energy)
        # counts += swings

        # Ensure counts are non-negative
        counts = np.maximum(counts, 0)

        # Plot if requested
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(energy, counts, label="Total Counts (Synthetic Data)")
            plt.plot(energy, measurements["Count"], label="Measurements", linestyle="--", alpha=0.7)
            plt.yscale("log")
            plt.fill_between(energy, noise, alpha=0.3, label="Baseline Noise")
            plt.scatter(energy[is_anomalous], counts[is_anomalous], color='red', label="Anomalous Regions", s=10)
            plt.title(f"Synthetic Spectrum with {num_peaks} Peaks")
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"tmp\\synthetic_{id_selected}.png")
            plt.close()

        # Return as DataFrame with clear components
        return pd.DataFrame({
            "Energy": energy,
            "Count": counts,
            "Noise": noise,
            "Gaussian": all_gaussians,
            "is_anomalous": is_anomalous
        })
