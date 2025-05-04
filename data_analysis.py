import seaborn as sns
import matplotlib.pyplot as plt
import src.measurements.api as mpi

dates = mpi.unique_dates()
measurement = mpi.measurement(dates)

measurement_diffs = (
    measurement.sort_values(by="energy").groupby("datetime").diff().dropna()
)
measurement_diffs["energy"] = measurement_diffs["energy"].round(2)
measurement_diffs = measurement_diffs.join(
    measurement, lsuffix="_l", rsuffix="_r", how="left"
)
diffs = measurement_diffs.groupby("energy_l").count().reset_index()
diffs["percent"] = diffs["count_l"] / diffs["count_l"].sum() * 100

plt.rcParams["svg.fonttype"] = "none"
fig = plt.figure(figsize=(7, 5))
ax = sns.barplot(
    diffs,
    x="energy_l",
    y="count_l",
    color="grey",
    linewidth=1.5,
    edgecolor=".5",
    facecolor=(0, 0, 0, 0),
)
labels = [
    f"{c / 1e3:.1f}K\n({p:.2f}%)" for c, p in zip(diffs["count_l"], diffs["percent"])
]
ax.bar_label(ax.containers[0], labels=labels, fontsize=10)
# plt.title("Untersuchung der Energiewerte")
plt.xlabel("Energiedifferenz [keV]")
plt.ylabel("Anzahl")
plt.grid(axis="y", alpha=0.2)
plt.yscale("log")
plt.ylim(1, 100000000)
plt.savefig("plots\\energy_diffs.svg")
plt.show()

# Extract counts and filter out zeros
counts = measurement["count"].to_numpy() + 1e-0
# counts = counts[counts > 0]

# Compute lognormal parameters
log_counts = np.log(counts)
sigma = np.std(log_counts)
mu = np.mean(log_counts)

# Create range for plotting
x = np.linspace(np.min(counts), np.max(counts), 500)
pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))

# Plot histogram + fitted lognormal PDF
sns.histplot(
    counts, bins=40, stat="density", kde=False, color="skyblue", log_scale=(True, False)
)
# plt.plot(x, pdf, 'r-', lw=2, label='Lognormal fit')
plt.xlabel("Count")
plt.ylabel("Density")
plt.legend()
plt.show()


measurement["count_shift"] = measurement["count"] + 1e-6
ax = sns.histplot(
    measurement,
    x="count_shift",
    log_scale=(True, False),
    bins=40,
    # kde=True,
    # kde_kws={"bw_adjust":10, "bw_method": "silverman"},
)
plt.show()

sns.lineplot(data=measurement_diffs, x="energy_l", y="energy_r")
plt.show()

measurement_diffs["energy_l"] = measurement_diffs["energy_l"].astype(str)
plt.figure(figsize=(10, 7))
ax = sns.histplot(
    measurement_diffs.loc[
        (measurement_diffs["energy_r"] > 0) & (measurement_diffs["count_r"] > 0)
    ],
    x="energy_r",
    y="count_r",
    bins=(2988, 40),
    log_scale=(False, True),
    cbar=True,
    cbar_kws={
        "orientation": "horizontal",
        "shrink": 1,
        "label": "Anzahl der Messwerte für die Energiewerte",
    },
    cmap=sns.color_palette("rocket_r", as_cmap=True),
    zorder=-10,
    # rasterized=True
)
cbar = ax.figure.axes[-1]
cbar.xaxis.label.set_size(14)
cbar.xaxis.labelpad = 15
cbar.tick_params(labelsize=12)
ax.set_rasterization_zorder(0)
plt.xlabel("Energie [keV]", size=14, labelpad=15)
plt.ylabel("Zählwert", size=14, labelpad=15)
plt.xticks(size=12)
plt.yticks(size=12)
plt.ylim(1)
plt.xlim(0, 2988)
plt.grid(alpha=0.2, which="both")
plt.savefig("plots\\count_energy_heatmap.svg", format="svg")
plt.show()

import numpy as np
from scipy.interpolate import interp1d


def interpolate_spectrum(energy_original, counts_original, energy_target):
    f = interp1d(
        energy_original,
        counts_original,
        kind="nearest",
        bounds_error=False,
        fill_value=0,
    )
    return f(energy_target)


step_size = measurement_diffs["energy_l"].mean()
energy_max = step_size * 8160
# etwas über 2987.8, MAX VALUE OF diffs
energy_axis = np.arange(0, energy_max, step_size)
test_energy = measurement.loc[measurement["datetime"] == dates[0]].sort_values(
    by="energy"
)
interpolated_counts = interpolate_spectrum(
    test_energy["energy"].values, test_energy["count"].values, energy_axis
)

test_energy["energy_axis"] = energy_axis
test_energy["interpolated_counts"] = interpolated_counts

plt.plot(energy_axis, interpolated_counts, color="red", alpha=0.5)
plt.plot(test_energy["energy"], test_energy["count"], alpha=0.5, color="black")
plt.xlim(0, 10)
plt.show()
