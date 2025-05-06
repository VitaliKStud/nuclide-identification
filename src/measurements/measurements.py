import os
import pandas as pd
import re
from sqlalchemy import text
import logging
from datetime import datetime
import src.measurements.api as mpi


class Measurements(mpi.API):
    def __init__(self):
        super().__init__()
        self.meta_data = {}
        self.measurements = pd.DataFrame([])
        self.pattern_time = r"Realtime:([\d.]+)Livetime:([\d.]+)"
        self.patter_coefs = r"([+-]?\d+\.\d+E[+-]?\d+)"

    def append_meta_data(
        self, date_from_file_name, coefficients, realtime, livetime, channels
    ):
        self.meta_data[date_from_file_name] = {
            "coefficients": coefficients,
            "realtime": realtime,
            "livetime": livetime,
            "channels": channels,
        }

    def process_single_file(self, filename):
        logging.info(f"Reading {self.path_measurements}{filename}")
        with open(f"{self.path_measurements}{filename}", "r") as f:
            # Converting Filename to Datetime
            date_from_file_name = datetime.strptime(
                filename.split("_")[0] + " " + filename.split("_")[1],
                "%Y-%m-%d %H-%M-%S",
            )

            # Getting All lines
            lines = [i.replace(",", ".") for i in f.readlines()]

            # Processing Meta-Data
            channels = int(lines[0].split(":")[-1].strip("\n "))
            match = re.search(
                self.pattern_time,
                lines[1].replace("\n", "").replace("\t", "").replace(" ", ""),
            )
            realtime = None
            livetime = None
            if match:
                realtime = float(match.group(1))
                livetime = float(match.group(2))
                logging.info(f"Realtime: {realtime}, Livetime: {livetime}")
            else:
                logging.info("Pattern not found!")

            all_polynom_coefs = lines[2].replace("\n", "").replace("channel", "")
            coefficients = [
                float(coef) for coef in re.findall(self.patter_coefs, all_polynom_coefs)
            ]
            self.append_meta_data(
                date_from_file_name, coefficients, realtime, livetime, channels
            )
            # Finish Processing MEta-Data

            # Processing Measurements
            data_rows = [
                i.replace("\n", "").replace(" ", "").split("\t") for i in lines[5:]
            ]
            data_rows = [
                (date_from_file_name, float(i[0]), int(i[1])) for i in data_rows
            ]
            # Finish Processing Measurements

            return data_rows

    def process_measurements_to_csv_to_db(self):
        """ """
        all_measurement_paths = [
            i for i in os.listdir(self.path_measurements) if ".txt" in i
        ]

        with self.engine.connect() as conn:
            with conn.begin():
                conn.execute(text('DROP TABLE IF EXISTS "measurements.measurements"'))

        for filename in all_measurement_paths:
            data_rows = self.process_single_file(filename)

            measurement = pd.DataFrame(
                data_rows, columns=["datetime", "energy", "count"]
            )
            logging.info(f"Writing {filename} to Database")
            measurement.to_sql(
                "measurements",
                self.engine,
                if_exists="append",
                index=False,
                schema="measurements",
            )
            logging.info(f"Wrote {filename} to Database")

            self.measurements = pd.concat([self.measurements, measurement], axis=0)

        self.measurements.to_csv(
            f"{self.path_output}measurements.csv", index_label="index"
        )
        logging.info(f"Saved {self.path_output}measurements.csv")

        meta_data_df = pd.DataFrame(self.meta_data).T.reset_index()
        meta_data_df[["coef_1", "coef_2", "coef_3", "coef_4"]] = pd.DataFrame(
            meta_data_df["coefficients"].tolist(), index=meta_data_df.index
        )
        meta_data_df.drop(columns=["coefficients"], inplace=True)
        meta_data_df = meta_data_df.rename(columns={"index": "datetime"})
        logging.info("Writing meta_data to Database")
        meta_data_df.to_sql(
            "meta_data", self.engine, if_exists="replace", index=False, schema="meta"
        )
        logging.info("Wrote meta_data to Database")

        meta_data_df.to_csv(f"{self.path_output}meta_data.csv", index=False)
        logging.info(f"Saved {self.path_output}meta_data.csv")

        files_combined_len = len(self.measurements["datetime"].unique())
        meta_data_len = len(self.meta_data)
        logging.info(
            f"Combined {files_combined_len} files. Meta-Data for {meta_data_len} files"
        )
