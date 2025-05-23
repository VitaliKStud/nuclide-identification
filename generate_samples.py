from src.generator.generator import Generator
import numpy as np
import pandas as pd
import random
import mlflow
import src.measurements.api as mpi
import torch
import matplotlib.pyplot as plt

dates = mpi.API().unique_dates()[0:6]
model = Generator().get_model()
min, max = Generator().get_min_max()

latent_space = []
for i in range(10):
    for date in dates:
        measurements = mpi.API().measurement([date])
        x = torch.tensor(measurements["count"].values, dtype=torch.float32)[:].squeeze()
        x[:] = (x[:] - min) / ( max - min + 1e-8 )
        x_it = x.to("cuda")
        x_hat, mean, logvar = model(x_it)
        z = model.reparameterize(mean, logvar).to("cpu").detach().numpy()
        x_hat_numpy = x_hat.to("cpu").detach().numpy()
        latent_space.append(z)

def generate_latent_space(latent_space=[], latent_idx=0, deepness=24):
    for i in range(501):
        data_to_generate = np.zeros(24, dtype="float32")
        for j in range(len(data_to_generate)):
            data_to_generate[j] = random.uniform(-3,3)
        # data_to_generate[latent_idx] = i/2
        latent_space.append(data_to_generate)
        # if latent_idx < deepness - 1:
        #     latent_space = generate_latent_space(latent_space=latent_space, latent_idx=latent_idx + 1)
    return latent_space

# latent_space = generate_latent_space()
df = pd.DataFrame(latent_space)
Generator().process(latent_space=latent_space)
