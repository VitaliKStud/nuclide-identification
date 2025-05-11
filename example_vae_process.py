from src.vae.training import Training
import src.peaks.api as ppi
import src.measurements.api as mpi
import uuid

dates = mpi.API().unique_dates()
dataset = (
    mpi.API()
    .measurement(dates[0:100])
    .sort_values(by=["datetime", "energy", "count"])
    .reset_index(drop=True)
)

Training(
    dataset=dataset,
    train_test_split=0.8,
    model_tag="vae testing mlflow workflow"
).vae_training()
