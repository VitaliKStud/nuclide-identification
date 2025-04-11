from tqdm import tqdm
import torch.nn.functional as F
import torch
from src.synthetics.hyperparameter import Hyperparameter
import matplotlib.pyplot as plt
from src.synthetics.vae import VAE
from torch.utils.data import DataLoader
from scipy.stats import norm
import numpy as np
from src.synthetics.cnn_vae import CNNVAe


class VAEModel:
    """
    Training and visualization of Variational Autoencoder.
    """

    def __init__(self, dataset, architecture="VAE"):
        self.dataset = dataset
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
            self.dataset, batch_size=Hyperparameter.BATCH_SIZE, shuffle=True
        )

    # Main Training Loop
    def train_vae(self):
        self.model.train()
        loss_history = []

        for epoch in range(Hyperparameter.NUM_EPOCHS):
            epoch_loss = 0
            for x in tqdm(
                self.data_loader, desc=f"Epoch {epoch}/{Hyperparameter.NUM_EPOCHS}"
            ):
                x = x.to(Hyperparameter.DEVICE)
                x_hat, mean, logvar = self.model(x)
                loss = self.vae_loss(x, x_hat, mean, logvar)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.data_loader.dataset)
            loss_history.append(avg_loss)

            # Save results periodically
            if (epoch + 1) % 250 == 0:
                for idx in range(
                    min(2, len(self.data_loader))
                ):  # Visualize first two samples
                    self.visualize_comparison(
                        x[idx].cpu().numpy(),
                        x_hat[idx].cpu().detach().numpy(),
                        epoch,
                        avg_loss,
                        name=f"sample_{idx}",
                    )
                self.visualize_latent_space_sampling(
                    x[idx].cpu().numpy(),
                    self.model,
                    device=Hyperparameter.DEVICE,
                    latent_dim=Hyperparameter.LATENT_DIM,
                    steps=10,
                    save_prefix=f"epoch_{epoch}",
                )

        return loss_history

    # Visualization Functions
    def visualize_comparison(
        self, original, reconstructed, epoch, loss, name="comparison"
    ):
        diff = reconstructed - original
        plt.figure(figsize=(10, 5))
        plt.plot(original, label="Original", linestyle="--", alpha=0.7)
        plt.plot(reconstructed, label="Reconstructed", alpha=0.7)
        plt.title(f"Epoch {epoch}, Loss: {loss:.4f}")
        plt.xlabel("Energy Index")
        plt.ylabel("Normalized Count")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"tmp/{name}_epoch_{epoch}_loss_{loss:.4f}.png")
        plt.close()

    def visualize_latent_space_sampling(
        self, original, model, device, latent_dim, steps=10, save_prefix="latent"
    ):
        z_samples = torch.linspace(-2, 2, steps)
        for dim in range(latent_dim):
            z = torch.zeros(steps, latent_dim).to(device)
            z[:, dim] = z_samples
            decoded = model.decode(z).cpu().detach().numpy()
            for i, signal in enumerate(decoded):
                if i > 3 or dim > 3:
                    pass
                else:
                    plt.figure(figsize=(10, 5))
                    plt.plot(original, label="Original", linestyle="--", alpha=0.7)
                    plt.plot(
                        signal, label=f"Latent Dim {dim}, Sample {i}", linestyle="--"
                    )
                    plt.title(f"Generated Signal - Dim {dim}, Sample {i}")
                    plt.grid()
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"tmp/{save_prefix}_dim{dim}_sample{i}.png")
                    plt.close()

    def vae_loss(self, x, x_hat, mean, logvar):
        reconstruction_loss = F.huber_loss(x_hat, x, reduction="sum")
        # plt.plot(np.arange(0, len(x_hat[0])), x_hat[0].cpu().detach().numpy(), color="red", label="reconstructed")
        # plt.plot(np.arange(0, len(x[0])), x[0].cpu().detach().numpy(), color="blue", label="original")
        # plt.show()
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence
