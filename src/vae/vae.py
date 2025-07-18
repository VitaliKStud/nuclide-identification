import torch
import torch.nn as nn
from config.loader import load_config
import logging


class VAE(nn.Module):
    """
    Variational Autoencoder to generate synthetic data.
    """

    def __init__(self):
        super().__init__()
        configs = load_config()
        self.device = configs["vae"]["device"]

        self.encoder = nn.Sequential(
            nn.Linear(configs["vae"]["input_dim"], configs["vae"]["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(
                configs["vae"]["hidden_dim"], configs["vae"]["second_hidden_dim"]
            ),
            nn.ReLU(),
            nn.Linear(
                configs["vae"]["second_hidden_dim"], configs["vae"]["latent_dim"]
            ),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(configs["vae"]["latent_dim"], configs["vae"]["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(configs["vae"]["hidden_dim"], configs["vae"]["input_dim"]),
            nn.Sigmoid(),
        )

        self.mean_layer = nn.Linear(
            configs["vae"]["latent_dim"], configs["vae"]["latent_dim"]
        )
        self.logvar_layer = nn.Linear(
            configs["vae"]["latent_dim"], configs["vae"]["latent_dim"]
        )

    def encode(self, x):
        x = self.encoder(x)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        logging.warning(f"z: {z}")
        return x_hat, mean, logvar
