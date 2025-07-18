import pandas as pd
from config.loader import load_engine
import src.nuclide.api as npi


class API:
    def __init__(self):
        self.engine = load_engine()
        self.npi = npi.API()

    def basic_statistics(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_basic_statistics",
            con=self.engine,
        )

    def found_isotopes_statistics(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_found_isotopes_statistics",
            con=self.engine,
        )

    def view_std_mean_min_max_statistics(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_std_mean_min_max_statistics",
            con=self.engine,
        )

    def view_dist_processed_measurements(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_dist_processed_measurements",
            con=self.engine,
        )

    def view_dist_processed_synthetics(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_dist_processed_synthetics",
            con=self.engine,
        )

    def view_dist_measurements(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_dist_measurements",
            con=self.engine,
        )

    def view_pm_isotopes_found(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_pm_isotopes_found",
            con=self.engine,
        )

    def view_re_pm_isotopes_found(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_re_pm_isotopes_found",
            con=self.engine,
        )

    def view_ps_isotopes_found(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_ps_isotopes_found",
            con=self.engine,
        )

    def view_re_ps_isotopes_found(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_re_ps_isotopes_found",
            con=self.engine,
        )

    def view_isotope_per_pm(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_isotope_per_pm",
            con=self.engine,
        )

    def view_isotope_per_re_pm(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_isotope_per_re_pm",
            con=self.engine,
        )

    def view_isotope_per_ps(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_isotope_per_ps",
            con=self.engine,
        )

    def view_isotope_per_re_ps(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_isotope_per_re_ps",
            con=self.engine,
        )

    def view_min_max(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_min_max",
            con=self.engine,
        )

    def view_identified_isotopes(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_identified_isotopes",
            con=self.engine,
        )

    def view_re_identified_isotopes(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_re_identified_isotopes",
            con=self.engine,
        )

    def view_processed_synthetics_latent_space(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_processed_synthetics_latent_space",
            con=self.engine,
        )

    def view_mean_measurement(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.view_mean_measurement",
            con=self.engine,
        )

    def processed_synthetics_latent_space(self):
        return pd.read_sql(
            sql="SELECT * FROM measurements.processed_synthetics_latent_space",
            con=self.engine,
        )
