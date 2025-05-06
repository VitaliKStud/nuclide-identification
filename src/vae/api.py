import pandas as pd
from config.loader import load_engine


class API:
    def __init__(self):
        self.engine = load_engine()

    def unique_dates(self):
        return pd.read_sql(
            sql='SELECT DISTINCT("datetime") FROM measurements.processed_synthetics',
            con=self.engine,
        )["datetime"].to_list()

    def synthetic(self, dates: list):
        if len(dates) == 1:
            query = f"SELECT * FROM measurements.processed_synthetics WHERE datetime = '{dates[0]}'"
        else:
            query = f'SELECT * FROM measurements.processed_synthetics WHERE "datetime" IN {tuple(dates)}'
        return (
            pd.read_sql(sql=query, con=self.engine)
            .sort_values(by=["datetime", "energy"])
            .reset_index(drop=True)
        )
