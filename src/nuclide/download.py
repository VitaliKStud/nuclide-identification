"""
This file contains all functionality to download nuclide-data from
https://www-nds.iaea.org/relnsd/vcharthtml/api_v0_guide.html

Complete pipeline -> DOWNLOAD DATA AS CSV FROM API -> SAVE TO DATABASE (POSTGRES)

EXAMPLE:
    Download().download_all_nuclides() # Will download all nuclides
"""

import requests
import logging
import os
import pandas as pd

from config import PATH, DB

logging.getLogger().setLevel(logging.INFO)

class Download:

    def __init__(self):
        self.ground_state_path = f"{PATH.OUTPUT}ground_state.csv"
        if not os.path.exists(PATH.OUTPUT):
            os.makedirs(PATH.OUTPUT)
        if not os.path.exists(PATH.NUCLIDES):
            os.makedirs(PATH.NUCLIDES)

    def get_ground_state(
            self,
            compromised=False
    ):
        """
        Will get all available nuclides from API.

        :param compromised: default = False, If true will return only nuclide-id, z, n and symbol as columns.
        :type compromised: bool
        :return: All available nuclides.
        :rtype: pd.DataFrame
        """

        if not os.path.exists(self.ground_state_path):
            logging.info(f"No {self.ground_state_path} found, downloading...")
            self._download_ground_state_file()
            logging.info(f"Loaded {self.ground_state_path}")
        ground_state = pd.read_csv(self.ground_state_path)
        if compromised is True:
            ground_state["nuclide_id"] = ground_state["symbol"] + (ground_state["z"] + ground_state["n"]).astype(str)
            ground_state = ground_state[["nuclide_id", "z", "n", "symbol"]]
        ground_state.to_sql("ground_state", DB.ENGINE, if_exists="replace", index=False, schema="nuclide")
        return ground_state

    def _download_single_nuclide(
            self,
            nuclide_id
    ):
        """
        Loading single nuclide data locally.
        """

        url = f"https://www-nds.iaea.org/relnsd/v1/data?fields=decay_rads&nuclides={nuclide_id}&rad_types=g"
        with requests.get(url, stream=True) as r:
            logging.info(f"LOADING: {nuclide_id}")
            if len(r.content) > 2:
                with open(f"{PATH.NUCLIDES}{nuclide_id}.csv", 'wb') as f:
                    for chunk in r.iter_content():
                        f.write(chunk)
                    logging.info(f"{nuclide_id} loaded and saved to {PATH.NUCLIDES}{nuclide_id}.csv")
            else:
                logging.warning(f"No data NUCLIDE-ID: {nuclide_id} found on API (probably not gamma)")

    def _download_ground_state_file(self):
        """
        Will load ground state file from API and save it to:

        - data\\ground_state_all_nuclides.csv
        """

        url = f"https://www-nds.iaea.org/relnsd/v1/data?fields=ground_states&nuclides=all"
        with requests.get(url, stream=True) as r:
            with open(self.ground_state_path, 'wb') as f:
                for chunk in r.iter_content():
                    f.write(chunk)
        logging.info(f"Ground-State for all nuclides loaded and saved to {self.ground_state_path}")

    def combine_nuclides_and_save_to_db(self):
        """
        Helper function to combine all nuclides into one large file.
        """

        all_files = [PATH.NUCLIDES + i for i in os.listdir(PATH.NUCLIDES) if ".csv" in i]
        combined_nuclides = pd.DataFrame([])
        logging.info("Combining all nuclides into one large file...")
        for file in all_files:
            single_nuclide = pd.read_csv(file)
            single_nuclide["nuclide_id"] = file.split("\\")[-1].replace(".csv", "").lower()
            combined_nuclides = pd.concat([combined_nuclides, single_nuclide], axis=0)
        combined_nuclides.to_csv(f"{PATH.OUTPUT}nuclides.csv", index_label="index")
        logging.info(f"SAVED: {PATH.OUTPUT}nuclides.csv")
        logging.info("Saving to Database...")
        combined_nuclides.to_sql("nuclide", DB.ENGINE, if_exists="replace", index=False, schema="nuclide")
        logging.info("Saved to Database")

    def download_all_nuclides(self):
        """
        Will load all nuclides from API. All data will be saved:

        - data/nuclides/...             # Singe files downloaded from the API
        - data/combined_nuclides.csv    # All files combined into one large file.
        """

        ground_state = self.get_ground_state(compromised=True)
        nuclides = ground_state["nuclide_id"].tolist()
        for nuclide in nuclides:
            if nuclide is not None and not os.path.exists(f"{PATH.NUCLIDES}{nuclide}.csv"):
                self._download_single_nuclide(nuclide_id=nuclide)
            else:
                logging.info(f"Found {nuclide} already downloaded, skipping.")
        self.combine_nuclides_and_save_to_db()
        logging.info(f"Loaded all nuclides, check {PATH.NUCLIDES}combined_nuclides.csv, as a merged file")
