import src.measurements.api as mpi
import src.statistics.api as spi
import src.peaks.api as ppi
from config.loader import load_config
from sqlalchemy import text
import pandas as pd
from config.loader import load_engine
import logging


class Splitter:
    """
    Initial splitting based on initial labeling. Especially for VAE and validation of VAE. Reserving CNN Training data.
    """
    def __init__(self):
        self.unique_dates = mpi.API().unique_dates()
        self.isotope_per_pm = spi.API().view_isotope_per_pm()
        self.engine = load_engine()
        self.vae_splitter = dict(load_config()["splitter"]["vae"])
        self.cnn_validation = dict(load_config()["splitter"]["cnn_validation"])
        self.max_isotopes_per_measurement = load_config()["splitter"][
            "max_isotopes_per_measurement"
        ]

    def split_keys(self):
        isotope_per_pm = spi.API().view_isotope_per_pm().sample(frac=1).reset_index(drop=True)
        splitted_keys = {
            "vae": {0: []},
            "cnn_validation": {0: []},
            "cnn_training": {0: []},
        }
        groups = isotope_per_pm.groupby("row_count")
        for counted_isotopes, group in groups:
            if counted_isotopes >= self.max_isotopes_per_measurement:
                pass
            else:
                number_to_split_vae = self.vae_splitter[str(counted_isotopes)]
                number_to_split_cnn_validation = self.cnn_validation[
                    str(counted_isotopes)
                ]
                ids_for_vae = isotope_per_pm["datetime"][
                    group.index[0:number_to_split_vae]
                ].tolist()
                ids_for_validation = isotope_per_pm["datetime"][
                    group.index[
                        number_to_split_vae : number_to_split_cnn_validation
                        + number_to_split_vae
                    ]
                ].tolist()
                ids_for_cnn_training = isotope_per_pm["datetime"][
                    group.index[number_to_split_cnn_validation + number_to_split_vae :]
                ].tolist()
                splitted_keys["vae"][0].extend(ids_for_vae)
                splitted_keys["cnn_validation"][0].extend(ids_for_validation)
                splitted_keys["cnn_training"][0].extend(ids_for_cnn_training)
        result = []
        for split_type, data in splitted_keys.items():
            for ts in data[0]:
                result.append({"type": split_type, "datetime": ts})
        splitted_data = pd.DataFrame(result)

        query = text("""
                     TRUNCATE TABLE measurements.splitted_keys_for_training_and_validation_pm
                     """)
        with self.engine.connect() as connection:
            try:
                with connection.begin():
                    connection.execute(query)
            except Exception as e:
                logging.warning(f"Deletion failed: {e}")
        splitted_data.to_sql(
            "splitted_keys_for_training_and_validation_pm",
            self.engine,
            if_exists="append",
            index=False,
            schema="measurements",
        )
