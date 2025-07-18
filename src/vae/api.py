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

        with self.engine.begin() as connection:
            data = pd.read_sql(sql=query, con=self.engine).sort_values(by=["datetime", "energy"]).reset_index(drop=True)
        return data

    def latent_space_shaps(self):

        query = f'SELECT * FROM measurements.latent_space_shaps'
        return (
            pd.read_sql(sql=query, con=self.engine)
            .sort_values(by=["datetime", "energy"])
            .reset_index(drop=True)
        )

    def re_unique_dates(self):
        try:
            return pd.read_sql(
                sql='SELECT DISTINCT("datetime") FROM measurements.re_processed_synthetics',
                con=self.engine,
            )["datetime"].to_list()
        except:
            return ["None"]

    def re_synhtetics(self, dates: list):
        if len(dates) == 1:
            query = f"SELECT * FROM measurements.re_processed_synthetics WHERE datetime = '{dates[0]}'"
        else:
            query = f'SELECT * FROM measurements.re_processed_synthetics WHERE "datetime" IN {tuple(dates)}'
        with self.engine.begin() as connection:
            data = pd.read_sql(sql=query, con=self.engine).sort_values(by=["datetime", "energy"]).reset_index(drop=True)
        return data