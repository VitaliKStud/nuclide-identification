from src.measurements import Measurements
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Dataset Class
class MeasurementsDataset(Dataset):
    def __init__(self, dataframe, group_size=8160):
        self.group_size = group_size
        self.data = torch.tensor(dataframe[["Energy", "Count"]].values, dtype=torch.float32)

        # Ensure the dataset size is divisible by group_size
        if len(self.data) % self.group_size != 0:
            raise ValueError("Dataset size must be divisible by group_size.")

        self.num_samples = len(self.data) // self.group_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.group_size
        end_idx = start_idx + self.group_size
        group = self.data[start_idx:end_idx]

        # Sort and normalize the data
        group = group[group[:, 0].argsort()]
        group[:, 1] = (group[:, 1] - group[:, 1].min()) / (group[:, 1].max() - group[:, 1].min() + 1e-8)

        return group[:, 1]  # Return only normalized "Count" column


# Variational Autoencoder Model
class VAE(nn.Module):
    def __init__(self, input_dim=8160, hidden_dim=1024, latent_dim=512, device="cuda"):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
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
        return x_hat, mean, logvar


# Loss Function
def vae_loss(x, x_hat, mean, logvar):
    reconstruction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence


# Visualization Functions
def visualize_comparison(original, reconstructed, epoch, loss, name="comparison"):
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


def visualize_latent_space_sampling(model, device, latent_dim, steps=10, save_prefix="latent"):
    z_samples = torch.linspace(-2, 2, steps)
    for dim in range(latent_dim):
        z = torch.zeros(steps, latent_dim).to(device)
        z[:, dim] = z_samples
        decoded = model.decode(z).cpu().detach().numpy()
        for i, signal in enumerate(decoded):
            plt.figure(figsize=(10, 5))
            plt.plot(signal, label=f"Latent Dim {dim}, Sample {i}", linestyle="--")
            plt.title(f"Generated Signal - Dim {dim}, Sample {i}")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"tmp/{save_prefix}_dim{dim}_sample{i}.png")
            plt.close()


# Main Training Loop
def train_vae(model, data_loader, optimizer, num_epochs, device):
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for x in tqdm(data_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            x = x.to(device)
            x_hat, mean, logvar = model(x)
            loss = vae_loss(x, x_hat, mean, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader.dataset)
        loss_history.append(avg_loss)

        # Save results periodically
        if epoch % 50 == 0:
            for idx in range(min(2, len(data_loader))):  # Visualize first two samples
                visualize_comparison(
                    x[idx].cpu().numpy(),
                    x_hat[idx].cpu().detach().numpy(),
                    epoch, avg_loss, name=f"sample_{idx}"
                )
            visualize_latent_space_sampling(model, device=DEVICE, latent_dim=LATENT_DIM, steps=10,
                                            save_prefix=f"epoch_{epoch}")

    return loss_history


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 8160
HIDDEN_DIM = 1024
LATENT_DIM = 16
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
NUM_EPOCHS = 500

# Load Measurements Data
measurements = Measurements().get_all_measurements()[0:81600]
dataset = MeasurementsDataset(measurements)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model Initialization
vae_model = VAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, device=DEVICE).to(DEVICE)
optimizer = torch.optim.Adam(vae_model.parameters(), lr=LEARNING_RATE)

# Train the Model
losses = train_vae(vae_model, data_loader, optimizer, NUM_EPOCHS, DEVICE)