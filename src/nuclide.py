import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import numpy as np
from config import PATH
import logging


class Nuclide:
    """
    This class is for handling Nuclide data. It will use the API to load nuclide data.
    """

    def __init__(
        self, nuclide_id: str = None, intensity_filter=None, unc_en_not_none=False
    ):
        self.nuclide_id = nuclide_id
        self.single_nuclide_path = f"{PATH.NUCLIDES}{self.nuclide_id}.csv"
        self.all_nuclides_path = f"{PATH.OUTPUT}ground_state_all_nuclides.csv"
        self.combined_nuclides_path = f"{PATH.OUTPUT}combined_nuclides.csv"
        if nuclide_id is not None:
            self._check_single_nuclide()
        else:
            self._check_all_nuclides()
        self.intensity_filter = intensity_filter
        self.unc_en_not_none = unc_en_not_none

    def _check_single_nuclide(self):
        """
        Checking if data is available for single nuclide. If not will download single nuclide-data.
        """

        if not os.path.isfile(self.single_nuclide_path):
            logging.info("No nuclide-data available locally, downloading...")
            self._download_single_nuclide()

    def _check_all_nuclides(self):
        """
        Checking if data is available for all nuclides. If not will download all nuclide-data.
        """

        if not os.path.isfile(self.combined_nuclides_path):
            logging.info("No data available locally, downloading...")
            self.download_all_nuclides()

    def _filter_nuclides(self, data):
        if self.intensity_filter is not None:
            data = data.loc[data["intensity"] > self.intensity_filter].reset_index(
                drop=True
            )
        if self.unc_en_not_none is True:
            data = data.loc[~data["unc_en"].isna()].reset_index(drop=True)
        return data

    def get_nuclides(self):
        """
        Will return all nuclides from the csv file (data/combined_nuclides.csv).

        :return: All nuclides.
        :rtype: pd.DataFrame
        """

        data = pd.read_csv(self.combined_nuclides_path, index_col="index")
        data = self._filter_nuclides(data)
        return data

    def get_nuclide(self):
        """
        Will return single nuclide from the csv file (data/nuclides/{NUCLIDE_ID}.csv).

        :return: Single nuclide.
        :rtype: pd.DataFrame
        """

        data = pd.read_csv(self.single_nuclide_path)
        data = self._filter_nuclides(data)

        return data

    def get_nuclide_distribution(self, step: float = 0.29):
        data = self.get_nuclide()
        distribution_data = None

        unique_energies = data["energy"].unique()
        bins = np.arange(-5, 5, 0.29)
        sigma = np.sqrt(1)
        mu = 0
        for unique_energy in unique_energies:
            s = (
                1
                / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-((bins - mu) ** 2) / (2 * sigma**2))
            )
            # distribution_data = pd.DataFrame([s, bins + unique_energy]).T.rename(columns={0: "Count", 1: "Energy"})

        return s

    # def identify_nuclides(self, data):

    def plot_peaks(self, filter_intensity: float = 0.0, save: bool = False):
        data = pd.read_csv(self.single_nuclide_path)
        fig = plt.figure(figsize=(15, 10))
        data = data.loc[data["intensity"] >= filter_intensity]
        data["energy"] = data["energy"].round(2)
        sns.barplot(data=data, x="energy", y="intensity", fill=False, color="black")
        plt.xlabel("Energy [keV]")
        plt.ylabel("Intensity [%]")
        plt.grid(alpha=0.2)
        plt.yticks(range(0, int(101), 10))
        if save is True:
            plt.savefig(f"plots\\peak_{self.nuclide_id}.png")
            plt.close()
        else:
            plt.show()
