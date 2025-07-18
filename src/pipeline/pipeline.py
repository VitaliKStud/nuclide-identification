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
import src.vae.api as vpi
from src.peaks.refinder import RePeakFinder
import src.nuclide.api as npi
from src.measurements.resplitter import ReSplitter


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
        self.nuclide_data = npi.API().nuclides(load_config()["repeakfinder"]["nuclides"])
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

    def train_cnn(self, use_relabled_data=False):
        cnn.Training(
            use_processed_synthetics=bool(
                load_config()["cnn"]["use_processed_synthetics"]
            ),
            use_processed_measuremnets=bool(
                load_config()["cnn"]["use_processed_measurements"],
            ),
            use_re_processed_data=use_relabled_data
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
            self, synthetic_prefix="", truncate_table=False, use_measurements_for_latent=False
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
        Generator().process(latent_space=latent_space, prefix=synthetic_prefix)

        query = text("""
                     CREATE OR REPLACE VIEW measurements.view_processed_synthetics_latent_space AS
                     SELECT grouped.datetime AS psl_datetime,
                            grouped.identified_isotopes,
                            grouped.row_count,
                            psl.*
                     FROM (SELECT ps.datetime,
                                  COUNT(DISTINCT ps.identified_isotope)                                          AS identified_isotopes,
                                  STRING_AGG(DISTINCT ps.identified_isotope, ','
                                             ORDER BY ps.identified_isotope)                                     AS row_count
                           FROM measurements.processed_synthetics ps
                           GROUP BY ps.datetime) grouped
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

    def find_measurements_peaks(self, measurement_peaks_prefix=""):
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
                measurement_peaks_prefix=measurement_peaks_prefix
            ).process_spectrum(return_detailed_view=False)

    def relable_measurements(self):
        query = text("""
                     TRUNCATE TABLE measurements.re_processed_measurements
                     """)
        with self.engine.connect() as connection:
            try:
                with connection.begin():
                    connection.execute(query)
            except Exception as e:
                logging.warning(f"Deletion failed: {e}")
        measurement_keys = ppi.API().unique_dates()
        len_keys = len(measurement_keys)
        processed = 0
        for i in range(0, len_keys, 500):
            batch = measurement_keys[i:i + 500]
            meas_data = ppi.API().measurement(batch)
            for measurement_key in batch:
                logging.warning(f"Measurement Key: {measurement_key}, {processed}/{len_keys}")
                filtered_data = meas_data.loc[meas_data["datetime"] == measurement_key].reset_index(drop=True)
                RePeakFinder(
                    selected_date=None,
                    data=filtered_data[["datetime", "energy", "count"]],
                    meta=None,
                    schema="re_processed_measurements",
                    matching_ratio=0,
                    interpolate_energy=False,
                    measurement_peaks_prefix="",
                    nuclide_data=self.nuclide_data
                ).process_spectrum(return_detailed_view=False)
                processed += 1

    def relable_synthetics(self):
        query = text("""
                     TRUNCATE TABLE measurements.re_processed_synthetics
                     """)
        with self.engine.connect() as connection:
            try:
                with connection.begin():
                    connection.execute(query)
            except Exception as e:
                logging.warning(f"Deletion failed: {e}")

        synthetic_keys = vpi.API().unique_dates()
        len_keys = len(synthetic_keys)
        processed = 0
        for i in range(0, len_keys, 500):
            batch = synthetic_keys[i:i + 500]
            synthetic_data = vpi.API().synthetic(batch)
            for synthetic_key in batch:
                logging.warning(f"Synthetic Key: {synthetic_key}, {processed}/{len_keys}")
                filtered_data = synthetic_data.loc[synthetic_data["datetime"] == synthetic_key].reset_index(drop=True)
                RePeakFinder(
                    selected_date=None,
                    data=filtered_data[["datetime", "energy", "count"]],
                    meta=None,
                    schema="re_processed_synthetics",
                    matching_ratio=0,
                    interpolate_energy=False,
                    measurement_peaks_prefix="",
                    nuclide_data=self.nuclide_data
                ).process_spectrum(return_detailed_view=False)
                processed += 1

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
            relable_measurements=False,
            relable_synthetics=False,
            synthetic_prefix="",
            measurement_peaks_prefix="",
            cnn_training=False,
            use_relabled_data=False,
            resplit_data=False,
    ):
        if download_nuclides is True:
            self.download_nuclides()
        if prepare_measurements is True:
            self.prepare_measurements()
        if find_measurements_peaks is True:
            self.find_measurements_peaks(measurement_peaks_prefix)
        if split_dataset is True:
            self.__split_dataset()
        if vae_training is True:
            self.train_vae()
        if generate_synthetics is True:
            self.generate_synthetics(synthetic_prefix, truncate_synhtetics, use_measurements_for_latent)
        if cnn_training is True:
            self.train_cnn(use_relabled_data)
        if relable_measurements is True:
            self.relable_measurements()
        if relable_synthetics is True:
            self.relable_synthetics()
        if resplit_data is True:
            ReSplitter().split_keys()
