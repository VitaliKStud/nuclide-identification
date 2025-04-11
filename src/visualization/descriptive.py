import seaborn as sns
import matplotlib.pyplot as plt
from src.measurements import Measurements


def plot_count_energy_heatmap():
    data = Measurements().get_measurements()
    data["Energy"] = data["Energy"].round(2)
    filtered_data = data.loc[(data["Energy"] > 0) & (data["Count"] > 0)]

    fig = plt.figure(figsize=(15, 10))
    ax = sns.histplot(
        filtered_data,
        x="Energy",
        y="Count",
        bins=(2988, 40),
        log_scale=(False, True),
        cbar=True,
        cbar_kws={
            "orientation": "horizontal",
            "shrink": 1,
            "label": "Anzahl der Messwerte für die Energiewerte",
        },
        cmap=sns.color_palette("rocket_r", as_cmap=True),
    )
    cbar = ax.figure.axes[-1]
    cbar.xaxis.label.set_size(14)
    cbar.xaxis.labelpad = 15
    cbar.tick_params(labelsize=12)
    plt.xlabel("Energie in keV", size=14, labelpad=15)
    plt.ylabel("Zählwert", size=14, labelpad=15)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.ylim(1)
    plt.xlim(0, 2988)
    plt.grid(alpha=0.2, which="both")
    plt.savefig("plots\\count_energy_heatmap.png")
    plt.close()
