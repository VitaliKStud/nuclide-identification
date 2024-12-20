import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests


class Nuclide:
    """
    This class is for handling Nuclide data. It will use the API to load nuclide data.
    """

    def __init__(self, nuclide_id: str = None):
        self.nuclide_id = nuclide_id
        self.single_nuclide_path = f'data\\nuclides\\{self.nuclide_id}.csv'
        self.all_nuclides_path = "data\\ground_state_all_nuclides.csv"
        self.combined_nuclides_path = "data\\combined_nuclides.csv"
        if nuclide_id is not None:
            self._check_single_nuclide()
        else:
            self._check_all_nuclides()

    def _check_single_nuclide(self):
        """
        Checking if data is available for single nuclide. If not will download single nuclide-data.
        """

        if not os.path.isfile(self.single_nuclide_path):
            print("No data available locally, downloading...")
            self._download_single_nuclide()

    def _check_all_nuclides(self):
        """
        Checking if data is available for all nuclides. If not will download all nuclide-data.
        """

        if not os.path.isfile(self.combined_nuclides_path):
            print("No data available locally, downloading...")
            self.download_all_nuclides()

    def _download_single_nuclide(self):
        """
        Loading single nuclide data locally.
        """

        if self.nuclide_id is not None:
            url = f"https://www-nds.iaea.org/relnsd/v1/data?fields=decay_rads&nuclides={self.nuclide_id}&rad_types=g"
            with requests.get(url, stream=True) as r:
                print(f"Content for: {self.nuclide_id} \n", r.content)
                if len(r.content) > 2:
                    with open(self.single_nuclide_path, 'wb') as f:
                        for chunk in r.iter_content():
                            f.write(chunk)
                        print(f"{self.nuclide_id} loaded and saved to {self.single_nuclide_path}")
                else:
                    print("No data found on API (probably not gamma)")

    def _combine_all_nuclides(self):
        """
        Helper function to combine all nuclides into one large file.
        """

        all_files = ["data\\nuclides\\" + i for i in os.listdir("data\\nuclides")]
        all_data = pd.DataFrame([])
        for file in all_files:
            data = pd.read_csv(file)
            data["nuclide_id"] = file.split("\\")[-1].replace(".csv", "").lower()
            all_data = pd.concat([all_data, data], axis=0)
        all_data.to_csv(self.combined_nuclides_path, index_label="index")

    def _download_ground_state_file(self):
        """
        Will load ground state file from API and save it to:

        - data\\ground_state_all_nuclides.csv
        """

        url = f"https://www-nds.iaea.org/relnsd/v1/data?fields=ground_states&nuclides=all"
        with requests.get(url, stream=True) as r:
            with open(self.all_nuclides_path, 'wb') as f:
                for chunk in r.iter_content():
                    f.write(chunk)
        print(f"Ground-State for all nuclides loaded and saved to {self.all_nuclides_path}")

    def download_all_nuclides(self):
        """
        Will load all nuclides from API. All data will be saved:

        - data/nuclides/...             # Singe files downloaded from the API
        - data/combined_nuclides.csv    # All files combined into one large file.
        """

        data = self.get_ground_state(compromised=True)
        nuclides = data["nuclide_id"].tolist()
        for nuclide in nuclides:
            Nuclide(nuclide)._download_single_nuclide()
        self._combine_all_nuclides()
        print("Loaded all nuclides, check data\\combined_nuclides.csv, as a merged file")

    def get_ground_state(self, compromised=False):
        """
        Will get all available nuclides from API.

        :param compromised: default = False, If true will return only nuclide-id, z, n and symbol as columns.
        :type compromised: bool
        :return: All available nuclides.
        :rtype: pd.DataFrame
        """

        if not os.path.exists(self.all_nuclides_path):
            print("No ground_state_all_nuclides.csv found, downloading...")
            self._download_ground_state_file()
        data = pd.read_csv(self.all_nuclides_path)
        if compromised is True:
            data["nuclide_id"] = data["symbol"] + (data["z"] + data["n"]).astype(str)
            data = data[["nuclide_id", "z", "n", "symbol"]]
        return data

    def get_nuclides(self):
        """
        Will return all nuclides from the csv file (data/combined_nuclides.csv).

        :return: All nuclides.
        :rtype: pd.DataFrame
        """
        return pd.read_csv(self.combined_nuclides_path, index_col="index")

    def get_nuclide(self):
        """
        Will return single nuclide from the csv file (data/nuclides/{NUCLIDE_ID}.csv).

        :return: Single nuclide.
        :rtype: pd.DataFrame
        """

        return pd.read_csv(self.single_nuclide_path)

    def get_nuclide_distribution(self, step: float = 0.29):
        pass

    def plot_peaks(self, filter_intensity: float = 0.0, save: bool = False):
        data = pd.read_csv(self.single_nuclide_path)
        fig = plt.figure(figsize=(15, 10))
        data = data.loc[data["intensity"] >= filter_intensity]
        data["energy"] = data["energy"].round(2)
        sns.barplot(data=data, x="energy", y="intensity", fill=False, color="black")
        plt.xlabel("Energy [keV]")
        plt.ylabel("Intensity [%]")
        plt.grid(alpha=0.2)
        plt.yticks(range(0, int(101), 10))
        if save is True:
            plt.savefig(f"plots\\peak_{self.nuclide_id}.png")
            plt.close()
        else:
            plt.show()
