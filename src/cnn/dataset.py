import torch
from torch.utils.data import Dataset


class MeasurementTraining(Dataset):
    """
    For dataset and dataloader for pytorch. Convert the dataset, so the DataLoader can be used.
    Normalize the dataset based on min/max across the entire dataset.
    """

    def __init__(self, dataset, datetimes, fitted_mlb):
        self.datetimes = datetimes

        self.fitted_mlb = fitted_mlb
        self.dataframe = dataset
        self.data_by_datetime = {
            dt: df.reset_index(drop=True)
            for dt, df in self.dataframe.groupby("datetime")
        }
        self.labels_by_datetime = {
            dt: self.fitted_mlb.transform(
                [df[df["identified_isotope"] != ""]["identified_isotope"].tolist()]
            )
            for dt, df in self.data_by_datetime.items()
        }
        self.processed_data_by_datetime = {}

        for dt, df in self.data_by_datetime.items():
            spec = df[["energy", "count"]].values
            spec = spec[spec[:, 0].argsort()]
            counts = spec[:, 1]
            counts = (counts - counts.min()) / (counts.max() - counts.min() + 1e-8)

            self.processed_data_by_datetime[dt] = torch.tensor(
                counts, dtype=torch.float32
            )

    def __len__(self):
        return len(self.datetimes)

    def __getmlbclasses__(self):
        return self.fitted_mlb.classes_

    def __get_scaler__(self):
        return "MinMaxScaler"

    def __getitem__(self, idx):
        selected_key = self.datetimes[idx]
        counts = self.processed_data_by_datetime[selected_key]
        labels = self.labels_by_datetime[selected_key]
        return counts, str(selected_key), labels
