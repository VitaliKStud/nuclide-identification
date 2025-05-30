import mlflow
import numpy as np
import torch
from config.loader import load_config
import os
import pandas as pd
import logging
from src.peaks.finder import PeakFinder


class Generator:
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
        self.model = self.__load_model()

    def get_model(self):
        return self.model

    def get_min_max(self):
        client = mlflow.tracking.MlflowClient(tracking_uri=load_config()["mlflow"]["uri"])
        run_id = client.get_latest_versions("VAE_CPU")[0].run_id
        run = client.get_run(run_id)
        min = float(run.data.params["min"])
        max = float(run.data.params["max"])
        return min, max

    def __load_model(self):
        os.environ["AWS_ACCESS_KEY_ID"] = load_config()["minio"]["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = load_config()["minio"]["AWS_SECRET_ACCESS_KEY"]
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = load_config()["minio"]["MLFLOW_S3_ENDPOINT_URL"]
        mlflow.set_tracking_uri(uri=self.model_uri)
        return mlflow.pytorch.load_model(f"models:/{self.model_name}/{self.model_version}").to(self.device)


    def __unscale(self, x_hat):
        min, max = self.get_min_max()
        return x_hat * (max - min) + min

    def process(self, latent_space):
        generator = self.generate(latent_space)
        sample_number = 0
        for energy_axis, generated_data in generator:
            try:
                synthetic_data = pd.DataFrame([])
                synthetic_data["energy"] = energy_axis
                synthetic_data["datetime"] = f"synthetic_{sample_number}"
                synthetic_data["count"] = generated_data
                PeakFinder(
                    selected_date=f"synthetic_{sample_number}",
                    data=synthetic_data,
                    meta=None,
                    schema="processed_synthetics",
                    nuclides=self.nuclides,
                    prominence=self.prominence,
                    tolerance=self.tolerance,
                    wlen=self.wlen,
                    nuclides_intensity=self.nuclide_intensity,
                    matching_ratio=self.matching_ratio,
                    interpolate_energy=False,
                ).process_spectrum(return_detailed_view=False)
                sample_number += 1
            except Exception as e:
                logging.warning(f"Could not process spectrum: {e}")

    def generate(self, latent_space):
        step_size = load_config()["measurements"]["step_size"]
        energy_max = step_size * load_config()["measurements"]["number_of_channels"]
        energy_axis = np.arange(0, energy_max, step_size)

        for z in latent_space:
            z_torch = torch.from_numpy(z).to(self.device)
            x_hat = self.model.decode(z_torch).to("cpu").detach().numpy()
            yield energy_axis, self.__unscale(x_hat)
