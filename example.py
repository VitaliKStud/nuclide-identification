from src.peaks.finder import PeakFinder
import seaborn as sns
import matplotlib.pyplot as plt

result = PeakFinder(
    selected_date="2017-04-13 13:36:00",
    nuclides=[
        "cs137",
        "co60",
        "i131",
        "tc99m",
        "ra226",
        "th232",
        "u238",
        "k40",
        "am241",
        "na22",
        "eu152",
        "eu154",
    ],
    nuclides_intensity=5,
).process_spectrum(return_detailed_view=False)

sns.lineplot(data=result, x="energy", y="count", color=sns.color_palette()[0])
sns.lineplot(data=result, x="energy", y="background", color=sns.color_palette()[1])
isotopes_identified = result.loc[result["peak"] is True]

colors = sns.color_palette("colorblind", n_colors=len(isotopes_identified) + 1)
for i, (_, row) in enumerate(isotopes_identified.iterrows()):
    plt.axvline(
        x=row["energy"],
        color=colors[i + 1],
        linestyle="--",
        label=f"{row['identified_isotope']} at {round(row['identified_peak'], 2)} keV",
    )

# Prevent duplicate labels in legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()
