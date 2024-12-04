import pandas as pd
import json
import seaborn as sns
import matplotlib.pylab as plt
from src.get_data_from_api import get_nuclide_data

get_nuclide_data("eu154")

# with open("data\\meta_data.json", "r") as f:
#     meta_data = json.load(f)
# data = pd.read_csv("data\\combined_data.csv", index_col="index")
# data_test = data[0:2*8160]
# fig = plt.figure(figsize=(20,10))
# sns.lineplot(data=data_test, x="Energy", y="Count", hue="ID_File")
# plt.xlim([300,600])
# plt.xscale("log")
# plt.yscale("log")
# plt.savefig("plots\\combined_data.png")

# data = pd.read_csv("eu154.csv")
# fig = plt.figure(figsize=(30,10))
# data = data.loc[data["intensity"] >= 1]
# sns.barplot(data=data, x="energy", y="intensity")
# plt.savefig("plots\\histo_eu154.png")


