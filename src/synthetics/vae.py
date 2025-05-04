import torch
import torch.nn as nn

from src.synthetics.hyperparameter import Hyperparameter


class VAE(nn.Module):
    """
    Variational Autoencoder to generate synthetic data.
    """

    def __init__(self, input_dim=8160, hidden_dim=1024, latent_dim=512, device="cuda"):
        super().__init__()
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

        self.decoder_energy = nn.Linear(latent_dim, input_dim)

        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)

    def encode(self, x):
        x = self.encoder(x)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        epsilon = torch.randn_like(logvar).to(Hyperparameter.DEVICE)  # sampling epsilon
        z = mean + logvar * epsilon
        return z

    def decode(self, z):
        return self.decoder(z)

    def decode_energy(self, z):
        # Decode energy from latent variable z
        return self.decoder_energy(z)

    def forward(self, x):
        x[:, 0].view(-1, Hyperparameter.INPUT_DIM)  # energy is the first column
        count = x[:, 1].view(-1, Hyperparameter.INPUT_DIM)

        mean, logvar = self.encode(count)
        z = self.reparameterize(mean, logvar)

        count_hat = self.decode(z)
        energy_hat = self.decode_energy(z)

        return energy_hat, count_hat, mean, logvar
