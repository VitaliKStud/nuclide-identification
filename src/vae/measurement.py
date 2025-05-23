import torch
from torch.utils.data import Dataset
from config.loader import load_config

# TODO PROBABLY IMPLEMENT OTHER SCALER (HERE MINMAX)

class Measurement(Dataset):
    """
    For dataset and dataloader for pytorch. Convert the dataset, so the DataLoader can be used.
    Normalize the dataset based on min/max across the entire dataset.
    """

    def __init__(
        self, dataframe, columns=["energy", "count", "datetime"]
    ):
        self.configs = load_config()
        self.group_size = load_config()["measurements"]["number_of_channels"]
        self.columns = columns
        self.data = torch.tensor(dataframe[["energy", "count"]].values, dtype=torch.float)
        self.datetimes = dataframe["datetime"].values
        self.data_min = self.data.min(dim=0).values
        self.data_max = self.data.max(dim=0).values
        if len(self.data) % self.group_size != 0:
            raise ValueError("Dataset size must be divisible by group_size.")
        self.num_samples = len(self.data) // self.group_size

    def __len__(self):
        return self.num_samples

    def __get_scaler__(self):
        return "MinMaxScaler"

    def __get_min_max__(self):
        return self.data_min[1], self.data_max[1]

    def __getitem__(self, idx):
        start_idx = idx * self.group_size
        end_idx = start_idx + self.group_size
        group = self.data[start_idx:end_idx]
        group_datetime = self.datetimes[start_idx:end_idx]
        group = group[group[:, 0].argsort()]  # Sort by energy
        group[:, 1] = (group[:, 1] - self.data_min[1]) / (
            self.data_max[1] - self.data_min[1] + 1e-8
        )
        return group[:, 1], str(group_datetime[0])
