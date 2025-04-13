import pandas as pd
from config import DB


def unique_dates():
    return pd.read_sql(
        sql='SELECT DISTINCT("datetime") FROM meta.meta_data', con=DB.ENGINE
    )["datetime"].to_list()


def measurement(dates):
    dates = [pd.Timestamp(t) for t in dates]
    timestamps_str = tuple(t.strftime("%Y-%m-%d %H:%M:%S") for t in dates)
    if len(timestamps_str) == 1:
        query = f"SELECT * FROM measurements.measurements WHERE datetime = '{timestamps_str[0]}'"
    else:
        query = f'SELECT * FROM measurements.measurements WHERE "datetime" IN {timestamps_str}'
    return (
        pd.read_sql(sql=query, con=DB.ENGINE)
        .sort_values(by="energy")
        .reset_index(drop=True)
    )


def meta_data(dates):
    dates = [pd.Timestamp(t) for t in dates]
    timestamps_str = tuple(t.strftime("%Y-%m-%d %H:%M:%S") for t in dates)
    if len(timestamps_str) == 1:
        query = f"SELECT * FROM meta.meta_data WHERE datetime = '{timestamps_str[0]}'"
    else:
        query = f'SELECT * FROM meta.meta_data WHERE "datetime" IN {timestamps_str}'
    return pd.read_sql(sql=query, con=DB.ENGINE)
