import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class CNNVAe(nn.Module):
    def __init__(self, input_length=3, h_dim=1024, z_dim=32, device="cuda"):
        super(CNNVAe, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 1024, kernel_size=4, stride=2),
            nn.ReLU(),
            # Flatten()
        )

        self.fc1 = nn.Linear(2038, h_dim)
        self.fc2 = nn.Linear(2038, h_dim)
        self.fc3 = nn.Linear(h_dim, 2038)

        self.decoder = nn.Sequential(
            # UnFlatten(),
            nn.ConvTranspose1d(1024, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar
