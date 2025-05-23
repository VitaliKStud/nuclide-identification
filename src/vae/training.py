import mlflow
import mlflow.pytorch
import mlflow.data.pandas_dataset
from tqdm import tqdm
import torch.nn.functional as F
import torch
from src.vae.vae import VAE
from torch.utils.data import DataLoader
from src.vae.measurement import Measurement
from config.loader import load_config
import logging
import os


class Training:

    def __init__(self, dataset, train_test_split, model_tag):
        self.configs = load_config()
        self.train_test_split = train_test_split
        self.raw_dataset = dataset
        self.model_tag = model_tag
        self.used_keys = self.raw_dataset["datetime"].unique().tolist()
        self.dataset = Measurement(self.raw_dataset)
        self.train_numbers = int(self.dataset.__len__() * self.train_test_split)
        self.test_numbers = self.dataset.__len__() - self.train_numbers
        self.train, self.test = torch.utils.data.random_split(
            dataset=self.dataset, lengths=[self.train_numbers, self.test_numbers],
            generator=torch.Generator().manual_seed(1)
        )
        self.model = VAE().to(self.configs["vae"]["device"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.configs["vae"]["learning_rate"]))
        self.train_loader = DataLoader(self.train, batch_size=self.configs["vae"]["batch_size"], shuffle=False)
        self.test_loader = DataLoader(self.test, batch_size=self.configs["vae"]["batch_size"], shuffle=False)
        self.training_loss_history = []
        self.training_reconstruction_loss_history = []
        self.training_kl_divergence_history = []

        self.validation_loss_history = []
        self.validation_reconstruction_loss_history = []
        self.validation_kl_divergence_history = []

        self.best_validation_loss = float("inf")
        self.best_model = None

    def __safe_model(self):
        os.environ["AWS_ACCESS_KEY_ID"] = load_config()["minio"]["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = load_config()["minio"]["AWS_SECRET_ACCESS_KEY"]
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = load_config()["minio"]["MLFLOW_S3_ENDPOINT_URL"]
        best_model = VAE().to(self.configs["vae"]["device"])
        best_model.load_state_dict(self.best_model)
        best_model.eval()
        used_data = mlflow.data.pandas_dataset.from_pandas(self.raw_dataset)
        mlflow.set_tracking_uri(uri=load_config()["mlflow"]["uri"])
        mlflow.set_registry_uri(uri=load_config()["mlflow"]["uri"])
        mlflow.set_experiment("SyntheticsVAE")

        with mlflow.start_run(run_name="VAE"):
            print(f"artifact_uri={mlflow.get_artifact_uri()}")
            mlflow.log_dict({"used_keys": [str(i) for i in self.used_keys]}, "artifacts.json")
            mlflow.log_text(self.raw_dataset.to_csv(), "dataset.csv")
            mlflow.log_param("architecture", "VAE")
            mlflow.log_param("input_dim", self.configs["vae"]["input_dim"])
            mlflow.log_param("hidden_dim", self.configs["vae"]["hidden_dim"])
            mlflow.log_param("latent_dim", self.configs["vae"]["latent_dim"])
            mlflow.log_param("epochs", self.configs["vae"]["epochs"])
            mlflow.log_param("batch_size", self.configs["vae"]["batch_size"])
            mlflow.log_param("learning_rate", self.configs["vae"]["learning_rate"])
            mlflow.log_param("device", self.configs["vae"]["device"])
            mlflow.log_param("scaler", self.dataset.__get_scaler__())
            mlflow.log_param("min", self.dataset.__get_min_max__()[0].detach().to("cpu").detach().numpy())
            mlflow.log_param("max", self.dataset.__get_min_max__()[1].to("cpu").detach().numpy())
            mlflow.log_param("best_validation_loss", self.best_validation_loss)
            mlflow.log_param("train_numbers", self.train_numbers)
            mlflow.log_param("test_numbers", self.test_numbers)
            mlflow.set_tag("Training Info", self.model_tag)
            for train_loss in self.training_loss_history:
                mlflow.log_metric("training_loss", train_loss)
            for val_loss in self.validation_loss_history:
                mlflow.log_metric("validation_loss", val_loss)
            for train_reconstruction_loss in self.training_reconstruction_loss_history:
                mlflow.log_metric("training_reconstruction_loss", train_reconstruction_loss)
            for val_reconstruction_loss in self.validation_reconstruction_loss_history:
                mlflow.log_metric("validation_reconstruction_loss", val_reconstruction_loss)
            for train_kl_divergence in self.training_kl_divergence_history:
                mlflow.log_metric("training_kl_divergence", train_kl_divergence)
            for val_kl_divergence in self.validation_kl_divergence_history:
                mlflow.log_metric("validation_kl_divergence", val_kl_divergence)

            mlflow.log_input(used_data, "used_dataset")
            mlflow.pytorch.log_model(best_model, "model_cuda")
            mlflow.pytorch.log_model(best_model.to("cpu"), "model_cpu")

    def __vae_validation(self):
        self.model.eval()
        validation_loss = 0
        validation_reconstruction_loss_count = 0
        validation_kl_divergence = 0
        with torch.no_grad():
            for i, (lead) in enumerate(self.test_loader):
                for i, x_it in enumerate(lead[0]):
                    datetime = lead[1][i]
                    x = x_it.to(self.configs["vae"]["device"])
                    count_hat, mean, logvar = self.model(x)
                    logging.warning(f"mean: {mean} logvar: {logvar}")
                    loss, reconstruction_loss_count, kl_divergence = self.__vae_loss(x, count_hat, mean, logvar)
                    validation_loss += loss.item()
                    validation_reconstruction_loss_count += reconstruction_loss_count.item()
                    validation_kl_divergence += kl_divergence.item()
        val_loss = validation_loss / self.test.__len__()
        val_reconstruction_loss_count = validation_reconstruction_loss_count / self.test.__len__()
        val_kl_divergence = validation_kl_divergence / self.test.__len__()

        if val_loss < self.best_validation_loss:
            self.best_validation_loss = val_loss
            self.best_model = self.model.state_dict()

        self.validation_reconstruction_loss_history.append(val_reconstruction_loss_count)
        self.validation_kl_divergence_history.append(val_kl_divergence)
        self.validation_loss_history.append(val_loss)

    def __vae_loss(self, x, count_hat, mean, logvar):
        reconstruction_loss_count = F.mse_loss(count_hat, x, reduction="sum")  # CROSS ENTROPY SHOULD BE TRIED OUT
        kl_divergence = -0.5 * torch.sum(1 + torch.log(logvar.pow(2)) - mean.pow(2) - logvar.pow(2))  # PARETO
        total_loss = reconstruction_loss_count + kl_divergence
        return total_loss, reconstruction_loss_count, kl_divergence

    def vae_training(self):
        for epoch in range(self.configs["vae"]["epochs"]):
            loop = tqdm(enumerate(self.train_loader))
            epoch_loss = 0
            epoch_reconstruction_loss_count = 0
            epoch_kl_divergence = 0
            for i, (lead) in loop:
                for i, x_it in enumerate(lead[0]):
                    datetime = lead[1][i]
                    x = x_it.to(self.configs["vae"]["device"])
                    count_hat, mean, logvar = self.model(x)
                    loss, reconstruction_loss_count, kl_divergence = self.__vae_loss(x, count_hat, mean, logvar)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    epoch_reconstruction_loss_count += reconstruction_loss_count.item()
                    epoch_kl_divergence += kl_divergence.item()
                    loop.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / self.train.__len__()
            self.training_reconstruction_loss_history.append(epoch_reconstruction_loss_count / self.train.__len__())
            self.training_kl_divergence_history.append(epoch_kl_divergence / self.train.__len__())
            self.training_loss_history.append(avg_loss)
            self.__vae_validation()
        self.__safe_model()
