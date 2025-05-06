import torch
from torch.utils.data import Dataset


# Dataset Class
class MeasurementsDataset(Dataset):
    """
    For dataset and dataloader for pytorch. Convert the dataset, so the DataLoader can be used.
    Normalize the dataset based on min/max across the entire dataset.
    """

    def __init__(
        self, dataframe, group_size=8160, columns=["energy", "count", "datetime"]
    ):
        self.group_size = group_size
        self.columns = columns
        self.data = torch.tensor(
            dataframe[["energy", "count"]].values, dtype=torch.float
        )
        self.datetimes = dataframe["datetime"].values

        # Compute the global min and max for normalization over the entire dataset
        self.data_min = self.data.min(dim=0).values  # Min values for energy and count
        self.data_max = self.data.max(dim=0).values  # Max values for energy and count

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
        group_datetime = self.datetimes[start_idx:end_idx]

        # Sort and normalize the data using the global min/max
        group = group[group[:, 0].argsort()]  # Sort by energy (first column)

        group[:, 1] = (group[:, 1] - self.data_min[1]) / (
            self.data_max[1] - self.data_min[1] + 1e-8
        )

        return group[:, 1], str(group_datetime[0])
