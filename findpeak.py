import pandas as pd
from scipy.signal import find_peaks
from src.measurements import Measurements
import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import electrocardiogram
import src.nuclide.api as npi
import src.measurements.api as mpi
from src.peaks.finder import PeakFinder
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

dates_ids = mpi.unique_dates()
measurement_example = mpi.measurement(dates=[dates_ids[0]]).sort_values(by="energy")
found_peaks, peakind = PeakFinder().find_possible_nuclides(
    data=measurement_example, prominence=500
)


fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=measurement_example["energy"],
        y=measurement_example["count"],
        mode="lines",
        name=f"File",
        zorder=10,
    )
)

for nuclide in found_peaks["nuclide_id"].unique():
    filtered_nuclide = found_peaks[found_peaks["nuclide_id"] == nuclide]
    fig.add_trace(
        go.Scatter(
            x=[filtered_nuclide["energy"].iloc[0], filtered_nuclide["energy"].iloc[0]],
            # Vertical line at each x position
            y=[0, 100000],  # Line from y=0 to y=intensity (with offset)
            mode="lines",
            name=f"Nuclide: {nuclide}",
            line=dict(width=2, color="red"),
            zorder=0,
        )
    )

for ind_peak in peakind:
    test = measurement_example.iloc[ind_peak]
    fig.add_trace(
        go.Scatter(
            x=[test["energy"], test["energy"]],
            # Vertical line at each x position
            y=[0, 100000],  # Line from y=0 to y=intensity (with offset)
            mode="lines",
            name=f"Nuclide: {ind_peak}",
            line=dict(width=2, color="black"),
            zorder=0,
        )
    )


fig.show()
