from src.synthetics.vae_model import VAEModel
import src.measurements.api as mpi
import src.peaks.api as ppi
import mlflow
from src.synthetics.hyperparameter import Hyperparameter
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config.loader import load_config
import pandas as pd
from src.peaks.finder import PeakFinder

generated_x_hat = []

for i in range(5):
    z = torch.from_numpy(np.arange(-1, 1, 1 / 12, dtype="float32")).to(
        Hyperparameter.DEVICE
    )
    z[0] = 0.0 + i / 1
    mlflow.set_tracking_uri(uri=load_config()["mlflow"]["uri"])
    model = mlflow.pytorch.load_model("runs:/278c374f3b39436e9648039d00374309/model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_hat = model.decode(z).to("cpu").detach().numpy()
    generated_x_hat.append(x_hat)

for i in generated_x_hat:
    plt.plot(i)
plt.show()



step_size = 0.34507313512321336
energy_max = step_size * 8160
energy_axis = np.arange(0, energy_max, step_size)

data = pd.DataFrame([])
for idx, i in enumerate(generated_x_hat):
    synthetic_data = pd.DataFrame(i)
    synthetic_data["energy"] = energy_axis
    synthetic_data["datetime"] = f"synthetic_{idx}"
    data = pd.concat([data, synthetic_data])
data = data.rename(columns={0: "count"})
sns.lineplot(data=data, x="energy", y="count", hue="datetime")

print(data.loc[data["datetime"] == "synthetic_1"].reset_index(drop=True))

data = data.loc[data["datetime"] == "synthetic_4"].reset_index(drop=True)
result = PeakFinder(
    selected_date="synthetic_4",
    data=data,
    meta=None,
    schema="processed_synthetics",
    nuclides=[
        "cs137",
        "co60",
        "i131",
        "tc99m",
        "ra226",
        "th232",
        "u238",
        "k40",
        "am241",
        "na22",
        "eu152",
        "eu154",
    ],
    prominence=0.2,
    tolerance=0.00,
    nuclides_intensity=10,
    matching_ratio=0,
    interpolate_energy=False,
).process_spectrum(return_detailed_view=False)
