import pandas as pd
from config import DB


def unique_dates():
    return pd.read_sql(
        sql='SELECT DISTINCT("datetime") FROM measurements.processed_measurements',
        con=DB.ENGINE,
    )["datetime"].to_list()


def measurement(dates: list):
    dates = [pd.Timestamp(t) for t in dates]
    timestamps_str = tuple(t.strftime("%Y-%m-%d %H:%M:%S") for t in dates)
    if len(timestamps_str) == 1:
        query = f"SELECT * FROM measurements.processed_measurements WHERE datetime = '{timestamps_str[0]}'"
    else:
        query = f'SELECT * FROM measurements.processed_measurements WHERE "datetime" IN {timestamps_str}'
    return (
        pd.read_sql(sql=query, con=DB.ENGINE)
        .sort_values(by=["datetime", "energy"])
        .reset_index(drop=True)
    )
