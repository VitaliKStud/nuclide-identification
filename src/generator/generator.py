import mlflow
import numpy as np
import torch
from config.loader import load_config
import os


class Generator:
    def __init__(self):
        self.device = load_config()["vae"]["device"]
        self.latent_dim = load_config()["vae"]["latent_dim"]
        self.model_name = "VAE_CPU"
        self.model_version = "latest"
        self.model_uri = load_config()["mlflow"]["uri"]
        self.model = self.__load_model()


    def __load_model(self):
        os.environ["AWS_ACCESS_KEY_ID"] = load_config()["minio"]["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = load_config()["minio"]["AWS_SECRET_ACCESS_KEY"]
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = load_config()["minio"]["MLFLOW_S3_ENDPOINT_URL"]
        mlflow.set_tracking_uri(uri=self.model_uri)
        return mlflow.pytorch.load_model(f"models:/{self.model_name}/{self.model_version}").to(self.device)

    def __unscale(self, x_hat):
        client = mlflow.tracking.MlflowClient(tracking_uri=load_config()["mlflow"]["uri"])
        run_id = client.get_latest_versions("VAE_CPU")[0].run_id
        run = client.get_run(run_id)
        min = float(run.data.params["min"])
        max = float(run.data.params["max"])
        return x_hat * (max - min) + min

    def generate(self, latent_space):
        # np.arange(-1, 1, 1 / (self.latent_dim / 2, dtype="float32")

        step_size = load_config()["measurements"]["step_size"]
        energy_max = step_size * load_config()["measurements"]["number_of_channels"]
        energy_axis = np.arange(0, energy_max, step_size)

        for z in latent_space:
            z_torch = torch.from_numpy(z).to(self.device)
            x_hat = self.model.decode(z_torch).to("cpu").detach().numpy()
            yield energy_axis, self.__unscale(x_hat)
