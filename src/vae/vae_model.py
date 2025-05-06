import mlflow
import mlflow.pytorch
from tqdm import tqdm
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from src.vae.vae import VAE
from torch.utils.data import DataLoader
from src.vae.measurement_dataset import MeasurementsDataset
from config.loader import load_config


class VAEModel:
    """
    Training and visualization of Variational Autoencoder.
    """

    def __init__(self, dataset):
        """
        Initializes the model, optimizer, and data loader based on the provided dataset and
        specifies the architecture of the model. If no architecture is specified, it defaults
        to "VAE". The model is initialized and moved to a predefined device, and the optimizer
        is configured with parameters of the model and a predetermined learning rate.
        Additionally, a data loader is created for the given dataset using a specific batch
        size and shuffle configuration.

        :param dataset: Input dataset to be used for training and evaluation.
        :type dataset: Any
        :param architecture: Specifies the architecture type of the model.
                             Can be either "VAE" or "CNNVAe". Defaults to "VAE".
        :type architecture: str
        """
        self.configs = load_config()
        self.dataset = MeasurementsDataset(dataset, group_size=8160)
        self.model = VAE().to(self.configs["vae"]["device"])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.configs["vae"]["learning_rate"]
        )
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.configs["vae"]["batch_size"], shuffle=False
        )

    # Main Training Loop
    def train_vae(self):
        self.model.train()
        loss_history = []

        for epoch in range(self.configs["vae"]["num_epochs"]):
            loop = tqdm(enumerate(self.data_loader))
            epoch_loss = 0
            for i, (lead) in loop:
                for i, x_it in enumerate(lead[0]):
                    datetime = lead[1][i]
                    x = x_it.to(self.configs["vae"]["device"])
                    count_hat, mean, logvar = self.model(x)
                    loss = self.vae_loss(x, count_hat, mean, logvar)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    loop.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / len(self.data_loader.dataset)
            loss_history.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                plt.plot(
                    count_hat.cpu().detach().numpy(),
                    label="Reconstructed",
                    alpha=0.7,
                )
                plt.savefig(f"tmp/without_x_epoch_{epoch}_loss_{avg_loss:.4f}_x.png")
                plt.close()

        self.model.eval()
        mlflow.set_tracking_uri(uri=load_config()["mlflow"]["uri"])
        mlflow.set_experiment("SyntheticsVAE")
        with mlflow.start_run(run_name="VAE"):
            mlflow.log_param("architecture", "VAE")
            mlflow.log_param("input_dim", self.configs["vae"]["input_dim"])
            mlflow.log_param("hidden_dim", self.configs["vae"]["hidden_dim"])
            mlflow.log_param("latent_dim", self.configs["vae"]["latent_dim"])
            mlflow.log_param("num_epochs", self.configs["vae"]["num_epochs"])
            mlflow.log_param("batch_size", self.configs["vae"]["batch_size"])
            mlflow.log_param("learning_rate", self.configs["vae"]["learning_rate"])
            mlflow.log_param("device", self.configs["vae"]["device"])
            mlflow.pytorch.log_model(self.model, "model")

        return loss_history

    def vae_loss(self, x, count_hat, mean, logvar):
        reconstruction_loss_count = F.mse_loss(count_hat, x, reduction="sum")
        kl_divergence = -0.5 * torch.sum(
            1 + torch.log(logvar.pow(2)) - mean.pow(2) - logvar.pow(2)
        )
        total_loss = reconstruction_loss_count + kl_divergence
        return total_loss
