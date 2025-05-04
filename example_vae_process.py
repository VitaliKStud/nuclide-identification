from src.synthetics.vae_model import VAEModel
import src.measurements.api as mpi

# mlflow.set_tracking_uri(uri=MLFLOW.URI)
# model = mlflow.pytorch.load_model("runs:/37612698db9f43b1900033c063984ce1/model")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()
#
# z_samples = torch.linspace(-2, 2, 10)
# for dim in range(Hyperparameter.LATENT_DIM):
#     z = torch.zeros(10, Hyperparameter.LATENT_DIM).to(device)
#     z[:, dim] = z_samples
#     decoded = model.decode(z).cpu().detach().numpy()
#     for i, signal in enumerate(decoded):
#         if i > 3 or dim > 3:
#             pass
#         else:
#             plt.figure(figsize=(10, 5))
#             plt.plot(
#                 signal, label=f"Latent Dim {dim}, Sample {i}", linestyle="--"
#             )
#             plt.title(f"Generated Signal - Dim {dim}, Sample {i}")
#             plt.grid()
#             plt.legend()
#             plt.tight_layout()
#             plt.savefig(f"tmp/_test_dim{dim}_sample{i}.png")
#             plt.close()

dates = mpi.unique_dates()
dataset = (
    mpi.measurement(dates[0:100])
    .sort_values(by=["datetime", "energy", "count"])
    .reset_index(drop=True)
)
VAEModel(
    dataset=dataset,
    architecture="VAE",
).train_vae()
