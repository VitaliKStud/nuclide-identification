import pandas as pd
from config.loader import load_config, load_engine


class API:
    """
    API for loading raw measurements and splitted keys
    """
    def __init__(self):
        self.engine = load_engine()
        self.path_measurements = load_config()["path"]["measurements"]
        self.path_nuclides = load_config()["path"]["nuclides"]
        self.path_output = load_config()["path"]["output"]

    def unique_dates(
        self,
    ):
        return pd.read_sql(
            sql='SELECT DISTINCT("datetime") FROM meta.meta_data', con=self.engine
        )["datetime"].to_list()

    def measurement(self, dates: list):
        dates = [pd.Timestamp(t) for t in dates]
        timestamps_str = tuple(t.strftime("%Y-%m-%d %H:%M:%S") for t in dates)
        if len(timestamps_str) == 1:
            query = f"SELECT * FROM measurements.measurements WHERE datetime = '{timestamps_str[0]}'"
        else:
            query = f'SELECT * FROM measurements.measurements WHERE "datetime" IN {timestamps_str}'
        return (
            pd.read_sql(sql=query, con=self.engine)
            .sort_values(by=["energy", "datetime"])
            .reset_index(drop=True)
        )

    def splitted_keys(self):
        query = (
            f"SELECT * FROM measurements.splitted_keys_for_training_and_validation_pm"
        )
        return pd.read_sql(sql=query, con=self.engine)

    def re_splitted_keys(self):
        query = (
            f"SELECT * FROM measurements.re_splitted_keys_for_training_and_validation_pm"
        )
        return pd.read_sql(sql=query, con=self.engine)

    def meta_data(self, dates):
        dates = [pd.Timestamp(t) for t in dates]
        timestamps_str = tuple(t.strftime("%Y-%m-%d %H:%M:%S") for t in dates)
        if len(timestamps_str) == 1:
            query = (
                f"SELECT * FROM meta.meta_data WHERE datetime = '{timestamps_str[0]}'"
            )
        else:
            query = f'SELECT * FROM meta.meta_data WHERE "datetime" IN {timestamps_str}'
        return pd.read_sql(sql=query, con=self.engine)
