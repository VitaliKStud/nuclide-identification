import pandas as pd
from config.loader import load_engine
import src.nuclide.api as npi


class API:
    def __init__(self):
        self.engine = load_engine()
        self.npi = npi.API()

    def basic_statistics(self):
        return pd.read_sql(
            sql='SELECT * FROM measurements.view_basic_statistics',
            con=self.engine,
        )

    def found_isotopes_statistics(self):
        return pd.read_sql(
            sql='SELECT * FROM measurements.view_found_isotopes_statistics',
            con=self.engine,
        )

    def view_std_mean_min_max_statistics(self):
        return pd.read_sql(
            sql='SELECT * FROM measurements.view_std_mean_min_max_statistics',
            con=self.engine,
        )

    def view_dist_processed_measurements(self):
        return pd.read_sql(
            sql='SELECT * FROM measurements.view_dist_processed_measurements',
            con=self.engine,
        )

    def view_dist_processed_synthetics(self):
        return pd.read_sql(
            sql='SELECT * FROM measurements.view_dist_processed_synthetics',
            con=self.engine,
        )

    def view_dist_measurements(self):
        return pd.read_sql(
            sql='SELECT * FROM measurements.view_dist_measurements',
            con=self.engine,
        )
