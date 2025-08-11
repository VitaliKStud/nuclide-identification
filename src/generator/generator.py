import mlflow
import numpy as np
import torch
from config.loader import load_config, load_engine
import os
import pandas as pd
import logging
from src.peaks.finder import PeakFinder
import src.measurements.api as mpi
import src.peaks.api as ppi


class Generator:
    """
    Generating synthetic data, after generation this will label included nuclides within synthetic data.
    """
    def __init__(self):
        self.device = load_config()["vae"]["device"]
        self.latent_dim = load_config()["vae"]["latent_dim"]
        self.model_name = "VAE_CPU"
        self.model_version = "latest"
        self.model_uri = load_config()["mlflow"]["uri"]
        self.tolerance = float(load_config()["peakfinder"]["tolerance"])
        self.nuclide_intensity = load_config()["peakfinder"]["nuclide_intensity"]
        self.matching_ratio = float(load_config()["peakfinder"]["matching_ratio"])
        self.prominence = int(load_config()["peakfinder"]["prominence"])
        self.nuclides = list(load_config()["peakfinder"]["nuclides"])
        self.wlen = int(load_config()["peakfinder"]["wlen"])
        self.rel_height = int(load_config()["peakfinder"]["rel_height"])
        self.width = int(load_config()["peakfinder"]["width"])
        self.model = self.__load_model()
        self.engine = load_engine()
        self.min, self.max = self.get_min_max()

    def get_model(self):
        return self.model

    def get_min_max(self):
        """
        Loading min max value for rescaling synthetic data
        """
        client = mlflow.tracking.MlflowClient(
            tracking_uri=load_config()["mlflow"]["uri"]
        )
        run_id = client.get_latest_versions("VAE_CPU")[0].run_id
        run = client.get_run(run_id)
        min = float(run.data.params["min"])
        max = float(run.data.params["max"])
        return min, max

    def __load_model(self):
        """
        Loading VAE model for generating synthetic data.
        """
        os.environ["AWS_ACCESS_KEY_ID"] = load_config()["minio"]["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = load_config()["minio"][
            "AWS_SECRET_ACCESS_KEY"
        ]
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = load_config()["minio"][
            "MLFLOW_S3_ENDPOINT_URL"
        ]
        mlflow.set_tracking_uri(uri=self.model_uri)
        logging.warning(f"Model: {self.model_name}, ModelVersion: {self.model_version}")
        return mlflow.pytorch.load_model(
            f"models:/{self.model_name}/{self.model_version}"
        ).to(self.device)

    def __unscale(self, x_hat):
        return x_hat * (self.max - self.min) + self.min

    def scale(self, x):
        return ((x - self.min) / (self.max - self.min)) + 1e-8

    def __generate_latent_space_from_measurements(self):
        """
        Generating latent space via measurements, by passing measurement spectra and generating latent space
        of it.
        """
        all_latents = []
        all_datetimes = []
        splitted_keys = mpi.API().splitted_keys()
        keys_for_generation = (
            splitted_keys.loc[splitted_keys["type"] == "cnn_training"]
            .reset_index(drop=True)["datetime"]
            .tolist()
        )
        measurements = ppi.API().measurement(keys_for_generation)
        for group in measurements.groupby("datetime"):
            datetime = group[0]
            measurement = group[1].sort_values(by="energy")[["energy", "count"]]
            for i in range(10): # Number of synthetics out of a measurement
                all_datetimes.append(str(datetime))
                x = self.scale(
                    torch.tensor(
                        measurement[["energy", "count"]].values, dtype=torch.float
                    )[:, 1].to(self.device)
                )
                mean, log_var = self.model.encode(x)
                latent_space = (
                    self.model.reparameterize(mean, log_var).to("cpu").detach().numpy()
                )
                if np.any(latent_space > 10) or np.any(latent_space < -10):
                    pass
                else:
                    all_latents.append(latent_space)
        return all_latents, all_datetimes

    def process(self, latent_space, prefix=""):
        """
        latent space can be passed to this function, if any latent space is used to generate data.
        Else it will generate latent space out of measurements and generate spectra. This function will
        also save the processed spectra into PostgreSQL and label nuclides within snythetic data here.
        """
        all_datetimes = None
        if latent_space is None:
            latent_space, all_datetimes = (
                self.__generate_latent_space_from_measurements()
            )
        generator = self.generate(latent_space)
        sample_number = 0
        for energy_axis, generated_data, latent_z in generator:
            try:
                if all_datetimes is not None:
                    datetime = all_datetimes[sample_number]
                else:
                    datetime = ""
                synthetic_data = pd.DataFrame([])
                synthetic_data["energy"] = energy_axis
                synthetic_data["datetime"] = f"{prefix}_synthetic_{sample_number}_{datetime}"
                synthetic_data["count"] = generated_data
                synthetic_data["datetime_from_measurement"] = datetime
                PeakFinder(
                    selected_date=f"synthetic_{sample_number}_{datetime}",
                    data=synthetic_data,
                    meta=None,
                    schema="processed_synthetics",
                    nuclides=self.nuclides,
                    prominence=self.prominence,
                    tolerance=self.tolerance,
                    width=self.width,
                    wlen=self.wlen,
                    rel_height=self.rel_height,
                    nuclides_intensity=self.nuclide_intensity,
                    matching_ratio=self.matching_ratio,
                    interpolate_energy=False,
                    measurement_peaks_prefix=prefix
                ).process_spectrum(return_detailed_view=False)

                columns = ["datetime", "datetime_from_measurement"] + [
                    i for i in range(len(latent_z))
                ]
                latent_data = pd.DataFrame(
                    [[f"{prefix}_synthetic_{sample_number}"] + [datetime] + list(latent_z)],
                    columns=columns,
                )
                latent_data.to_sql(
                    "processed_synthetics_latent_space",
                    self.engine,
                    if_exists="append",
                    index=False,
                    schema="measurements",
                )
                sample_number += 1
            except Exception as e:
                logging.warning(f"Could not process spectrum: {e}")

    def generate(self, latent_space):
        """
        Encoding out of latent space vector and interpolating dependend energy for generated spectra
        """
        step_size = load_config()["measurements"]["step_size"]
        energy_max = step_size * load_config()["measurements"]["number_of_channels"]
        energy_axis = np.arange(0, energy_max, step_size)

        for z in latent_space:
            z_torch = torch.from_numpy(z).to(self.device)
            x_hat = self.model.decode(z_torch).to("cpu").detach().numpy()
            yield energy_axis, self.__unscale(x_hat), z
