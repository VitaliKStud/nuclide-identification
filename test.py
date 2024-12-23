import src as ni
from src import synthetics

measurements = ni.Measurements().get_measurements()[0:81600]
dataset = synthetics.MeasurementsDataset(measurements)

VAE = synthetics.VAEModel(dataset=dataset)
losses = VAE.train_vae()