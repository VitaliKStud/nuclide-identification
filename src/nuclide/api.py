import pandas as pd
from config.loader import load_config, load_engine


class API:
    def __init__(self):
        self.engine = load_engine()
        self.path_measurements = load_config()["path"]["measurements"]
        self.path_nuclides = load_config()["path"]["nuclides"]
        self.path_output = load_config()["path"]["output"]

    def unique_nuclides(self):
        return pd.read_sql(
            sql='SELECT DISTINCT("nuclide_id") FROM nuclide.nuclide', con=self.engine
        )["nuclide_id"].to_list()

    def nuclides(self, nuclide_ids="all", intensity=0):
        if len(nuclide_ids) > 1 and nuclide_ids != "all":
            nuclide_ids = tuple(nuc_id for nuc_id in nuclide_ids)
            query = f"""
            SELECT 
            "nuclide_id", "energy", "intensity", "d_z", "d_n"
            FROM nuclide.nuclide
            WHERE "nuclide_id" IN {nuclide_ids}
            AND "intensity" IS NOT NULL 
    --         AND "unc_en" IS NOT NULL
            AND "intensity" > {intensity}
            """
        elif len(nuclide_ids) == 1 and nuclide_ids != "all":
            nuclide_ids = tuple(nuc_id for nuc_id in nuclide_ids)
            query = f"""
            SELECT 
            "nuclide_id", "energy", "intensity", "d_z", "d_n"
            FROM nuclide.nuclide
            WHERE "nuclide_id" = '{nuclide_ids[0]}'
            AND"intensity" IS NOT NULL 
    --         AND "unc_en" IS NOT NULL
            AND "intensity" > {intensity}
            """
        elif nuclide_ids == "all":
            query = f"""
            SELECT 
            "nuclide_id", "energy", "intensity", "d_z", "d_n"
            FROM nuclide.nuclide
            WHERE "intensity" IS NOT NULL 
    --         AND "unc_en" IS NOT NULL
            AND "intensity" > {intensity}
            """
        else:
            return None
        return pd.read_sql(sql=query, con=self.engine)

    def nuclides_max_intensity(self):
        query = """
        SELECT DISTINCT ON ("nuclide_id") 
            "nuclide_id", 
            "energy", 
            "intensity", 
            "d_z", 
            "d_n"
        FROM nuclide.nuclide
        WHERE "intensity" IS NOT NULL
        ORDER BY "nuclide_id", "intensity" DESC;
        """
        return pd.read_sql(sql=query, con=self.engine)

    def nuclides_by_intensity(self, intensity):
        query = f"""
        SELECT
            "nuclide_id", 
            "energy", 
            "intensity", 
            "d_z", 
            "d_n"
        FROM nuclide.nuclide
        WHERE "intensity" IS NOT NULL 
        AND "unc_en" IS NOT NULL
        AND "intensity" > {intensity}
        ORDER BY "nuclide_id", "intensity" DESC
        """
        return pd.read_sql(sql=query, con=self.engine)
