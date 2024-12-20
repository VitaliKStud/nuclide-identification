import pandas as pd
import json
import seaborn as sns
import matplotlib.pylab as plt
# from src.nuclide_dashboard import app
#
# app.run_server(debug=True, use_reloader=True, host='127.0.0.1', port=5432)

# get_nuclide_data("eu154")

import numpy as np
import matplotlib.pyplot as plt
from src.nuclide import Nuclide

data = Nuclide("am241").get_nuclide_data()
data = data.loc[data["intensity"] > 1]

# Parameter definieren

peaks = data["energy"].tolist()
intensities = data["intensity"].tolist()
detector_resolution = 1
energy_range = np.linspace(0, 2000, 3000)

# Hintergrundmodell
background = 5 * np.exp(-0.1 * energy_range)

# Funktion zur Gauß-Verteilung
def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Peaks generieren
spectrum = np.zeros_like(energy_range)
for peak, intensity in zip(peaks, intensities):
    fwhm = detector_resolution * (peak / max(peaks))  # Energieabhängige FWHM
    sigma = fwhm / 2.355  # Umrechnung FWHM -> Standardabweichung
    peak_distribution = intensity * gaussian(energy_range, peak, sigma)
    spectrum += peak_distribution

# Untergrund hinzufügen
spectrum += background

# Zufälliges Rauschen hinzufügen
noise = np.random.normal(0, 0.5, size=energy_range.shape)
spectrum += noise

# Plot
plt.plot(energy_range, spectrum, label="Simuliertes Spektrum")
plt.xlabel("Energie (keV)")
plt.ylabel("Zählrate")
plt.legend()
plt.show()

