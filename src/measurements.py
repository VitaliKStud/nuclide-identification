import os
import pandas as pd
import re
import json


class Measurements:

    def __init__(self):
        pass

    def convert_data_to_csv_json(self, path="data\\measurements"):
        """
        :param path: default:="data\\measurements"
        :type path: str

        Structure of a file should be:

        .. code-block:: text

            number of channels:  8160
            Realtime:   113963,06   Livetime:   114063,12
            -4,355441E-1+2,904070E-1*channel+5,651753E-8*channel^2+0,000000E+0*channel^3

            energy in keV	counts
               -0,44	      27
               -0,15	      82
                0,15	     135
                0,44	     185
                0,73	     269
                1,02	     505
                ...          ...

        Reading all .txt files and converting them to json and csv.
        Combining all .txt Results and saving meta-data for every measurement.

        Two files will be created:

        - data/meta_data.json
        - data/combined_measurements.csv.

        """
        all_paths = ["data\\" + i for i in os.listdir(path) if ".txt" in i]
        pattern_time = r"Realtime:([\d.]+)Livetime:([\d.]+)"
        patter_coefs = r"([+-]?\d+\.\d+E[+-]?\d+)"

        meta_data = {}
        all_data = pd.DataFrame([])
        for path in all_paths:
            with open(path, "r") as f:
                id_file = path.split("\\")[-1]
                all_lines = [i.replace(",", ".") for i in f.readlines()]

                channels = int(all_lines[0].split(":")[-1].strip("\n "))

                match = re.search(pattern_time, all_lines[1].replace("\n", "")
                                  .replace("\t", "").replace(" ", ""))
                realtime = None
                livetime = None
                if match:
                    realtime = float(match.group(1))
                    livetime = float(match.group(2))
                    print(f"Realtime: {realtime}, Livetime: {livetime}")
                else:
                    print("Pattern not found!")

                all_polynom_coefs = all_lines[2].replace("\n", "").replace("channel", "")

                coefficients = [float(coef) for coef in re.findall(patter_coefs, all_polynom_coefs)]

                data_rows = [i.replace("\n", "")
                             .replace(" ", "").split("\t") for i in all_lines[5:]]
                data_rows = [(id_file, float(i[0]), int(i[1])) for i in data_rows]

            meta_data[id_file] = {"coefficients": coefficients, "realtime": realtime, "livetime": livetime,
                                  "channels": channels}
            data = pd.DataFrame(data_rows, columns=["ID_File", "Energy", "Count"])

            all_data = pd.concat([all_data, data], axis=0)

        all_data.to_csv("data\\combined_measurements.csv", index_label="index")


        with open("data\\meta_data.json", "w") as f:
            json.dump(meta_data, f)

        files_combined_len = len(all_data["ID_File"].unique())
        meta_data_len = len(meta_data)
        print(f"Combined {files_combined_len} files. Meta-Data for {meta_data_len} files")

    @staticmethod
    def get_measurements():
        """
        Loads Measurements.

        :return: All Measurements combined as a DataFrame
        """
        return pd.read_csv("data\\combined_measurements.csv")


