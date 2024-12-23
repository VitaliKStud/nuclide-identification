import torch
from torch.utils.data import Dataset

# Dataset Class
class MeasurementsDataset(Dataset):
    """
    For dataset and dataloader for pytorch. Convert the dataset, so the DataLoader can be used.
    """

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