import pandas as pd
from config.loader import load_engine
import src.nuclide.api as npi


class API:
    """
    API for loading generated, labeled data.
    """
    def __init__(self):
        self.engine = load_engine()
        self.npi = npi.API()

    def unique_keys(self):
        return pd.read_sql(
            sql='SELECT DISTINCT("datetime") FROM measurements.processed_synthetics',
            con=self.engine,
        )["datetime"].to_list()

    def synthetics(self, keys: list):
        keys_str = tuple(keys)
        if len(keys_str) == 1:
            query = f"SELECT * FROM measurements.processed_synthetics WHERE datetime = '{keys_str[0]}'"
        else:
            query = f'SELECT * FROM measurements.processed_synthetics WHERE "datetime" IN {keys_str}'
        return (
            pd.read_sql(sql=query, con=self.engine)
            .sort_values(by=["datetime", "energy"])
            .reset_index(drop=True)
        )

    def synthetics_for_meas(self, keys: list):
        keys_str = tuple(keys)
        if len(keys_str) == 1:
            query = f"SELECT * FROM measurements.processed_synthetics WHERE datetime_from_measurement = '{keys_str[0]}'"
        else:
            query = f'SELECT * FROM measurements.processed_synthetics WHERE "datetime_from_measurement" IN {keys_str}'
        return (
            pd.read_sql(sql=query, con=self.engine)
            .sort_values(by=["datetime", "energy"])
            .reset_index(drop=True)
        )


