from src.vae.training import Training
import src.measurements.api as mpi
from src.nuclide.download import Download
from src.peaks.finder import PeakFinder
from src.measurements.measurements import Measurements
import src.peaks.api as ppi
from src.generator.generator import Generator
import numpy as np
import random
from config.loader import load_config, load_engine
from src.measurements.splitter import Splitter
import src.cnn.training as cnn
import logging
from sqlalchemy import text


class Pipeline:
    def __init__(self):
        self.tolerance = float(load_config()["peakfinder"]["tolerance"])
        self.nuclide_intensity = load_config()["peakfinder"]["nuclide_intensity"]
        self.matching_ratio = float(load_config()["peakfinder"]["matching_ratio"])
        self.interpolate_energy = bool(
            load_config()["peakfinder"]["interpolate_energy"]
        )
        self.prominence = int(load_config()["peakfinder"]["prominence"])
        self.nuclides = list(load_config()["peakfinder"]["nuclides"])
        self.wlen = int(load_config()["peakfinder"]["wlen"])
        self.width = int(load_config()["peakfinder"]["width"])
        self.rel_height = int(load_config()["peakfinder"]["rel_height"])
        self.engine = load_engine()

    def train_vae(self):
        all_keys = mpi.API().splitted_keys()
        keys_vae = all_keys.loc[all_keys["type"] == "vae"]["datetime"].tolist()
        dataset = (
            ppi.API()
            .measurement(keys_vae)
            .sort_values(by=["datetime", "energy", "count"])
            .reset_index(drop=True)
        )

        Training(dataset=dataset, train_test_split=0.8, model_tag="VAE").vae_training()

    def train_cnn(self):
        cnn.Training(
            use_processed_synthetics=bool(
                load_config()["cnn"]["use_processed_synthetics"]
            ),
            use_processed_measuremnets=bool(
                load_config()["cnn"]["use_processed_measurements"]
            ),
        ).cnn_training()

    def download_nuclides(self):
        Download().download_all_nuclides()

    def __generate_latent_space(self, latent_space=[]):
        latent_dim = load_config()["vae"]["latent_dim"]
        min_space = load_config()["generator"]["min_space"]
        max_space = load_config()["generator"]["max_space"]
        for i in range(load_config()["generator"]["number_of_samples"]):
            data_to_generate = np.zeros(latent_dim, dtype="float32")
            for j in range(len(data_to_generate)):
                data_to_generate[j] = random.uniform(min_space, max_space)
            latent_space.append(data_to_generate)
        return latent_space

    def __split_dataset(self):
        Splitter().split_keys()

    def generate_synthetics(
        self, truncate_table=False, use_measurements_for_latent=False
    ):
        if truncate_table is True:
            query = text("""
                         TRUNCATE TABLE measurements.processed_synthetics
                         """)
            with self.engine.connect() as connection:
                try:
                    with connection.begin():
                        connection.execute(query)
                except Exception as e:
                    logging.warning(f"Deletion failed: {e}")

            query = text("""
                         DROP VIEW measurements.view_processed_synthetics_latent_space
                         """)
            with self.engine.connect() as connection:
                try:
                    with connection.begin():
                        connection.execute(query)
                except Exception as e:
                    logging.warning(f"Deletion failed: {e}")

            query = text("""
                         DROP TABLE measurements.processed_synthetics_latent_space
                         """)
            with self.engine.connect() as connection:
                try:
                    with connection.begin():
                        connection.execute(query)
                except Exception as e:
                    logging.warning(f"Deletion failed: {e}")
        if use_measurements_for_latent is True:
            latent_space = None
        else:
            latent_space = self.__generate_latent_space()
        Generator().process(latent_space=latent_space)

        query = text("""
        CREATE OR REPLACE VIEW measurements.view_processed_synthetics_latent_space AS
        SELECT 
            grouped.datetime AS psl_datetime,
            grouped.identified_isotopes,
            grouped.row_count,
            psl.*
        FROM (
            SELECT 
                ps.datetime,
                COUNT(DISTINCT ps.identified_isotope) AS identified_isotopes,
                STRING_AGG(DISTINCT ps.identified_isotope, ',' ORDER BY ps.identified_isotope) AS row_count
            FROM measurements.processed_synthetics ps
            GROUP BY ps.datetime
        ) grouped
        JOIN measurements.processed_synthetics_latent_space psl 
          ON grouped.datetime = psl.datetime
        ORDER BY grouped.datetime;
        """)
        with self.engine.connect() as connection:
            try:
                with connection.begin():
                    connection.execute(query)
            except Exception as e:
                logging.warning(f"View Creation Failed: {e}")

    def prepare_measurements(self):
        Measurements().process_measurements_to_csv_to_db()

    def find_measurements_peaks(self, schema="processed_measurements"):
        dates = mpi.API().unique_dates()

        for date in dates:
            PeakFinder(
                selected_date=date,
                data=mpi.API().measurement([date]),
                meta=mpi.API().meta_data([date]),
                schema="processed_measurements",
                nuclides=self.nuclides,
                prominence=self.prominence,
                tolerance=self.tolerance,
                width=self.width,
                rel_height=self.rel_height,
                wlen=self.wlen,
                nuclides_intensity=self.nuclide_intensity,
                matching_ratio=self.matching_ratio,
                interpolate_energy=self.interpolate_energy,
            ).process_spectrum(return_detailed_view=False)

    def run(
        self,
        download_nuclides=False,
        prepare_measurements=False,
        find_measurements_peaks=False,
        split_dataset=False,
        vae_training=False,
        generate_synthetics=False,
        truncate_synhtetics=False,
        use_measurements_for_latent=False,
        cnn_training=False,
    ):
        if download_nuclides is True:
            self.download_nuclides()
        if prepare_measurements is True:
            self.prepare_measurements()
        if find_measurements_peaks is True:
            self.find_measurements_peaks()
        if split_dataset is True:
            self.__split_dataset()
        if vae_training is True:
            self.train_vae()
        if generate_synthetics is True:
            self.generate_synthetics(truncate_synhtetics, use_measurements_for_latent)
        if cnn_training is True:
            self.train_cnn()
