from src.vae.training import Training
import src.measurements.api as mpi
from src.nuclide.download import Download
from src.peaks.finder import PeakFinder
from src.measurements.measurements import Measurements
import src.peaks.api as ppi
from src.generator.generator import Generator
import numpy as np
import random
from config.loader import load_config


class Pipeline:
    def __init__(self):
        self.tolerance = float(load_config()["peakfinder"]["tolerance"])
        self.nuclide_intensity = load_config()["peakfinder"]["nuclide_intensity"]
        self.matching_ratio = float(load_config()["peakfinder"]["matching_ratio"])
        self.interpolate_energy = bool(load_config()["peakfinder"]["interpolate_energy"])
        self.prominence = int(load_config()["peakfinder"]["prominence"])

    def train_vae(self):
        dates = ppi.API().unique_dates()
        dataset = (
            ppi.API()
            .measurement(dates[0:100])
            .sort_values(by=["datetime", "energy", "count"])
            .reset_index(drop=True)
        )

        Training(
            dataset=dataset,
            train_test_split=0.8,
            model_tag="VAE"
        ).vae_training()

    def download_nuclides(self):
        Download().download_all_nuclides()

    def __generate_latent_space(self, latent_space=[]):
        for i in range(501):
            data_to_generate = np.zeros(24, dtype="float32")
            for j in range(len(data_to_generate)):
                data_to_generate[j] = random.uniform(-3, 3)
            latent_space.append(data_to_generate)
        return latent_space

    def generate_synthetics(self):
        latent_space = self.__generate_latent_space()
        Generator().process(latent_space=latent_space)

    def prepare_measurements(self):
        Measurements().process_measurements_to_csv_to_db()
        self.__identify_peaks()

    def __identify_peaks(self, schema="processed_measurements"):
        dates = mpi.API().unique_dates()

        for date in dates:
            PeakFinder(
                selected_date=date,
                data=mpi.API().measurement([date]),
                meta=mpi.API().meta_data([date]),
                schema="processed_measurements",
                nuclides=[
                    "cs137",
                    "co60",
                    "i131",
                    "tc99m",
                    "ra226",
                    "th232",
                    "u238",
                    "k40",
                    "am241",
                    "na22",
                    "eu152",
                    "eu154",
                ],
                prominence=self.prominence,
                tolerance=self.tolerance,
                nuclides_intensity=self.nuclide_intensity,
                matching_ratio=self.matching_ratio,
                interpolate_energy=self.interpolate_energy,
            ).process_spectrum(return_detailed_view=False)

    def run(self,
            download_nuclides=False,
            prepare_measurements=False,
            vae_training=False,
            generate_synthetics=False,
            ):
        if download_nuclides is True:
            self.download_nuclides()
        if prepare_measurements is True:
            self.prepare_measurements()
        if vae_training is True:
            self.train_vae()
        if generate_synthetics is True:
            self.generate_synthetics()
