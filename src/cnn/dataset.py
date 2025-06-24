import torch
from torch.utils.data import Dataset, IterableDataset
from config.loader import load_config, load_engine
import src.peaks.api as ppi
import src.statistics.api as spi
import src.measurements.api as mpi
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


# TODO PROBABLY IMPLEMENT OTHER SCALER (HERE MINMAX)


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
            # Normalize
            # counts = (counts - counts.mean()) /(counts.std() + 1e-8)
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


class MeasurementValidation(Dataset):
    """
    For dataset and dataloader for pytorch. Convert the dataset, so the DataLoader can be used.
    Normalize the dataset based on min/max across the entire dataset.
    """

    def __init__(self, dataset, datetimes, isotopes, fitted_mlb):
        self.configs = load_config()
        self.group_size = load_config()["measurements"]["number_of_channels"]
        self.engine = load_engine()
        self.statistics = spi.API().view_min_max()

        self.data_min = (
            self.statistics.loc[
                self.statistics["source_table"] == "processed_measurements_cnn_training"
            ]
            .reset_index(drop=True)["min"]
            .values[0]
        )
        self.data_max = (
            self.statistics.loc[
                self.statistics["source_table"] == "processed_measurements_cnn_training"
            ]
            .reset_index(drop=True)["max"]
            .values[0]
        )
        self.splitted_keys = mpi.API().splitted_keys()
        self.datetimes = (
            self.splitted_keys.loc[self.splitted_keys["type"] == "cnn_validation"]
            .reset_index(drop=True)["datetime"]
            .tolist()
        )

        self.isotopes = spi.API().view_identified_isotopes()
        self.isos = self.isotopes.loc[
            (self.isotopes["identified_isotopes"] != "")
            & (self.isotopes["source_table"] == "processed_measurements")
        ]["identified_isotopes"].tolist()
        self.unique_isos = list(set(self.isos))
        self.base_classes = [s.split(",") for s in self.isos]
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(self.base_classes)
        self.ppi = ppi.API()

    def __len__(self):
        return len(self.datetimes)

    def __getmlbclasses__(self):
        return self.mlb.classes_

    def __get_scaler__(self):
        return "MinMaxScaler"

    def __get_min_max__(self):
        return self.data_min, self.data_max

    def __getitem__(self, idx):
        selected_key = self.datetimes[idx]
        dataframe = self.ppi.measurement([selected_key])
        found_isotopes = [
            dataframe.loc[dataframe["identified_isotope"] != ""]
            .reset_index(drop=True)["identified_isotope"]
            .tolist()
        ]
        isotopes_in_data = self.mlb.transform(found_isotopes)

        data = torch.tensor(dataframe[["energy", "count"]].values, dtype=torch.float)
        group = data[data[:, 0].argsort()]
        group[:, 1] = (group[:, 1] - min(group[:, 1])) / (
            max(group[:, 1]) - min(group[:, 1]) + 1e-8
        )
        return group[:, 1], str(selected_key), isotopes_in_data
