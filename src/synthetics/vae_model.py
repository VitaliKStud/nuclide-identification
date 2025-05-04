import mlflow
import mlflow.pytorch
from tqdm import tqdm
import torch.nn.functional as F
import torch
from src.synthetics.hyperparameter import Hyperparameter
import matplotlib.pyplot as plt
from src.synthetics.vae import VAE
from torch.utils.data import DataLoader
from src.synthetics.cnn_vae import CNNVAe
from src.synthetics.measurement_dataset import MeasurementsDataset
from config import MLFLOW


class VAEModel:
    """
    Training and visualization of Variational Autoencoder.
    """

    def __init__(self, dataset, architecture="VAE"):
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
        self.dataset = MeasurementsDataset(dataset, group_size=8160)
        if architecture == "VAE":
            self.model = VAE(
                input_dim=Hyperparameter.INPUT_DIM,
                hidden_dim=Hyperparameter.HIDDEN_DIM,
                latent_dim=Hyperparameter.LATENT_DIM,
                device=Hyperparameter.DEVICE,
            ).to(Hyperparameter.DEVICE)
        elif architecture == "CNNVAe":
            self.model = CNNVAe(
                input_length=Hyperparameter.INPUT_DIM,
                z_dim=Hyperparameter.LATENT_DIM,
                device=Hyperparameter.DEVICE,
            ).to(Hyperparameter.DEVICE)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=Hyperparameter.LEARNING_RATE
        )
        self.data_loader = DataLoader(
            self.dataset, batch_size=Hyperparameter.BATCH_SIZE, shuffle=False
        )

    # Main Training Loop
    def train_vae(self):
        self.model.train()
        loss_history = []

        for epoch in range(Hyperparameter.NUM_EPOCHS):
            loop = tqdm(enumerate(self.data_loader))
            epoch_loss = 0
            for i, (lead) in loop:
                for i, x_it in enumerate(lead[0]):
                    lead[1][i]
                    x = x_it[0:,].to(Hyperparameter.DEVICE)
                    # x = x.view(-1, Hyperparameter.INPUT_DIM)
                    energy_hat, count_hat, mean, logvar = self.model(x)

                    loss = self.vae_loss(x, energy_hat, count_hat, mean, logvar)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    loop.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / len(self.data_loader.dataset)
            loss_history.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                plt.plot(
                    energy_hat.cpu().detach().numpy()[0],
                    count_hat.cpu().detach().numpy()[0],
                    label="Reconstructed",
                    alpha=0.7,
                )
                plt.savefig(f"tmp/epoch_{epoch}_loss_{avg_loss:.4f}_x.png")
                plt.close()
                plt.plot(
                    count_hat.cpu().detach().numpy()[0],
                    label="Reconstructed",
                    alpha=0.7,
                )
                plt.savefig(f"tmp/without_x_epoch_{epoch}_loss_{avg_loss:.4f}_x.png")
                plt.close()

            # # Save results periodically
            # if (epoch + 1) % 50 == 0:
            #     for idx in range(min(2, len(self.data_loader))):  # Visualize first two samples
            #         self.visualize_comparison(
            #             x.cpu().numpy()[0],
            #             x_hat.cpu().detach().numpy()[0],
            #             epoch,
            #             avg_loss,
            #             name=f"sample_{idx}",
            #         )
            #     self.visualize_latent_space_sampling(
            #         x.cpu().numpy()[0],
            #         self.model,
            #         device=Hyperparameter.DEVICE,
            #         latent_dim=Hyperparameter.LATENT_DIM,
            #         steps=10,
            #         save_prefix=f"epoch_{epoch}",
            #     )

        self.model.eval()
        mlflow.set_tracking_uri(uri=MLFLOW.URI)
        mlflow.set_experiment("SyntheticsVAE")
        with mlflow.start_run(run_name="VAE"):
            mlflow.log_param("architecture", "VAE")
            mlflow.log_param("input_dim", Hyperparameter.INPUT_DIM)
            mlflow.log_param("hidden_dim", Hyperparameter.HIDDEN_DIM)
            mlflow.log_param("latent_dim", Hyperparameter.LATENT_DIM)
            mlflow.log_param("num_epochs", Hyperparameter.NUM_EPOCHS)
            mlflow.log_param("batch_size", Hyperparameter.BATCH_SIZE)
            mlflow.log_param("learning_rate", Hyperparameter.LEARNING_RATE)
            mlflow.log_param("device", Hyperparameter.DEVICE)
            mlflow.pytorch.log_model(self.model, "model")

        return loss_history

    # # Visualization Functions
    # def visualize_comparison(
    #     self, original, reconstructed, epoch, loss, name="comparison"
    # ):
    #     # reconstructed - original
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(original, label="Original", linestyle="--", alpha=0.7)
    #     plt.plot(reconstructed, label="Reconstructed", alpha=0.7)
    #     plt.title(f"Epoch {epoch}, Loss: {loss:.4f}")
    #     plt.xlabel("Energy Index")
    #     plt.ylabel("Normalized Count")
    #     plt.legend()
    #     plt.grid()
    #     plt.tight_layout()
    #     plt.savefig(f"tmp/{name}_epoch_{epoch}_loss_{loss:.4f}.png")
    #     plt.close()
    #
    # def visualize_latent_space_sampling(
    #     self, original, model, device, latent_dim, steps=10, save_prefix="latent"
    # ):
    #     z_samples = torch.linspace(-2, 2, steps)
    #     for dim in range(latent_dim):
    #         z = torch.zeros(steps, latent_dim).to(device)
    #         z[:, dim] = z_samples
    #         decoded = model.decode(z).cpu().detach().numpy()
    #         for i, signal in enumerate(decoded):
    #             if i > 3 or dim > 3:
    #                 pass
    #             else:
    #                 plt.figure(figsize=(10, 5))
    #                 plt.plot(original, label="Original", linestyle="--", alpha=0.7)
    #                 plt.plot(
    #                     signal, label=f"Latent Dim {dim}, Sample {i}", linestyle="--"
    #                 )
    #                 plt.title(f"Generated Signal - Dim {dim}, Sample {i}")
    #                 plt.grid()
    #                 plt.legend()
    #                 plt.tight_layout()
    #                 plt.savefig(f"tmp/{save_prefix}_dim{dim}_sample{i}.png")
    #                 plt.close()

    def vae_loss(self, x, energy_hat, count_hat, mean, logvar):
        # Assuming x has two columns: energy and count
        count = x[:, 1]  # count is the second column
        energy = x[:, 0]  # energy is the first column

        # Reconstruction loss for count (use MSE loss)
        reconstruction_loss_count = F.mse_loss(count_hat, count, reduction="sum")

        # Optional: Reconstruction loss for energy (use MSE loss, but linear)
        reconstruction_loss_energy = F.mse_loss(energy_hat, energy, reduction="sum")

        # KL divergence
        kl_divergence = -0.5 * torch.sum(
            1 + torch.log(logvar.pow(2)) - mean.pow(2) - logvar.pow(2)
        )

        # Total loss (reconstruction loss + KL divergence)
        total_loss = (
            reconstruction_loss_count + kl_divergence + reconstruction_loss_energy
        )
        return total_loss
