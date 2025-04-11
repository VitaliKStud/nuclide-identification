import pandas as pd
from scipy.signal import find_peaks, find_peaks_cwt
import src.nuclide.api as npi
import numpy as np


class PeakFinder:
    def __init__(self):
        pass

    def find_peaks_in_data(self, data=None, distance=700, prominence=150):
        count = data["count"].to_numpy()
        energy = data["energy"].to_numpy()
        peaks, _ = find_peaks(count, distance=distance, prominence=prominence)
        peakind = find_peaks_cwt(count, np.arange(1, len(count) + 1), noise_perc=20)
        energy_peaks = energy[peaks]
        return peaks, energy_peaks, peakind

    def find_possible_nuclides(
        self, data=None, range_for_diff=0.5, distance=None, prominence=150
    ):
        peaks, energy_peaks, peakind = self.find_peaks_in_data(
            data=data, distance=distance, prominence=prominence
        )
        possible_nuclides = pd.DataFrame()
        nuclides = npi.nuclides_by_intensity(10)
        for idx, energy_peak in enumerate(energy_peaks):
            nuclides["diff"] = abs(nuclides["energy"] - energy_peak).round(3)
            nuclides = nuclides.sort_values(by="diff").reset_index(drop=True)
            possible_nuclide = nuclides[nuclides["diff"] <= range_for_diff]
            if possible_nuclide.empty:
                pass
            else:
                possible_nuclide = pd.DataFrame(possible_nuclide.iloc[0]).T
                possible_nuclide["EnergyPeakRef"] = energy_peak
                possible_nuclide["CountRef"] = peaks[idx]
                possible_nuclides = pd.concat(
                    [possible_nuclides, possible_nuclide], axis=0
                )
        if possible_nuclides.empty:
            return None
        else:
            return possible_nuclides.drop_duplicates(keep="first").reset_index(
                drop=True
            ), peakind
