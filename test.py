from platform import architecture

import src as ni
from src import synthetics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from src.measurements import Measurements
import numpy as np
from matplotlib.ticker import LogLocator
from torch.utils.data import DataLoader
from src.synthetics.hyperparameter import Hyperparameter


# data_generator = synthetics.SyntheticDataGenerator()
# combined_data = pd.DataFrame([])
# for i in range(0,10):
#     data = data_generator.generate_synthetic_data(plot=True)
#     combined_data = pd.concat([combined_data, data], axis=0)
# dataset = synthetics.MeasurementsDataset(combined_data, columns=["Energy", "Count", "is_anomalous"])

real_data = synthetics.MeasurementsDataset(Measurements().get_measurements()[0:81600], columns=["Energy", "Count"])
VAE = synthetics.VAEModel(dataset=real_data, architecture="VAE")
losses = VAE.train_vae()

model = VAE.model
data_loader = DataLoader(real_data, batch_size=Hyperparameter.BATCH_SIZE, shuffle=True)

counter = 0
for i in data_loader:
    for j in i:
        if counter > 10:
            pass
        else:
            reconst = model(j.to("cuda"))[0].cpu().detach().numpy()

            plt.plot(np.arange(0, len(reconst)), reconst, color="red", label="reconstructed")
            plt.plot(np.arange(0, len(reconst)), j.detach().numpy(), color="blue", label="original")
            plt.legend()
            # plt.yscale("log")
            plt.savefig(f"tmp\\{counter}.png")
            plt.close()

            counter += 1




