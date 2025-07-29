import pandas as pd
from config.loader import load_engine
import src.nuclide.api as npi


class API:
    def __init__(self):
        self.engine = load_engine()
        self.npi = npi.API()

    def unique_dates(self):
        return pd.read_sql(
            sql='SELECT DISTINCT("datetime") FROM measurements.processed_measurements',
            con=self.engine,
        )["datetime"].to_list()

    def re_unique_dates(self):
        return pd.read_sql(
            sql='SELECT DISTINCT("datetime") FROM measurements.re_processed_measurements',
            con=self.engine,
        )["datetime"].to_list()

    def measurement(self, dates: list):
        dates = [pd.Timestamp(t) for t in dates]
        timestamps_str = tuple(t.strftime("%Y-%m-%d %H:%M:%S") for t in dates)
        if len(timestamps_str) == 1:
            query = f"SELECT * FROM measurements.processed_measurements WHERE datetime = '{timestamps_str[0]}'"
        else:
            query = f'SELECT * FROM measurements.processed_measurements WHERE "datetime" IN {timestamps_str}'
        with self.engine.begin() as connection:
            data = pd.read_sql(sql=query, con=self.engine).sort_values(by=["datetime", "energy"]).reset_index(drop=True)
        return data

    def re_measurement(self, dates: list):
        dates = [pd.Timestamp(t) for t in dates]
        timestamps_str = tuple(t.strftime("%Y-%m-%d %H:%M:%S") for t in dates)
        if len(timestamps_str) == 1:
            query = f"SELECT * FROM measurements.re_processed_measurements WHERE datetime = '{timestamps_str[0]}'"
        else:
            query = f'SELECT * FROM measurements.re_processed_measurements WHERE "datetime" IN {timestamps_str}'
        with self.engine.begin() as connection:
            data = pd.read_sql(sql=query, con=self.engine).sort_values(by=["datetime", "energy"]).reset_index(drop=True)
        return data
