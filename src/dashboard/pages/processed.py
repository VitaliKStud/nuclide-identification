from dash import dcc, html, Input, Output, callback, register_page, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import src.statistics.api as spi

register_page(__name__, "/processed", title="ProcessedData")

# BASIC STATISTICS
basic_statistics = spi.API().basic_statistics()
basic_statistics_processed_measurements = basic_statistics.loc[
    basic_statistics["source_table"] == "processed_measurements"
    ].reset_index(drop=True)
basic_statistics_measurements = basic_statistics.loc[
    basic_statistics["source_table"] == "measurements"
    ].reset_index(drop=True)
basic_statistics_processed_synthetics = basic_statistics.loc[
    basic_statistics["source_table"] == "processed_synthetics"
    ].reset_index(drop=True)
relativ_processed_measurements = round((basic_statistics_processed_measurements["total_measurements"] /
                                        basic_statistics_measurements["total_measurements"]) * 100, 2)

# MEAN MAX STD STATISTICS
std_mean_min_max_statistics = spi.API().view_std_mean_min_max_statistics()

# FOUND ISOTOPES
found_isotopes_statistics = spi.API().found_isotopes_statistics()

# DISTRIBUTIONS
dist_processed_measurements = spi.API().view_dist_processed_measurements()
dist_processed_synthetics = spi.API().view_dist_processed_synthetics()
dist_measurements = spi.API().view_dist_measurements()


isotopes_dist_processed_fig = px.histogram(
    found_isotopes_statistics.loc[found_isotopes_statistics["source_table"] == "processed_measurements"],
    x='isotopes',
    y='count_processed_measurements',
    title="Processed Measurements",
    color="source_table"
)
isotopes_dist_synthetics_fig = px.histogram(
    found_isotopes_statistics.loc[found_isotopes_statistics["source_table"] == "processed_synthetics"],
    x='isotopes',
    y='count_processed_measurements',
    title="Processed Synthetics",
    color="source_table",
)
count_dist_processed_measurements = px.histogram(
    dist_processed_measurements,
    x='log_bin',
    y='frequency',
    nbins=40,
    title="Processed Measurements Distributions Counts",
)

count_dist_processed_synthetics = px.histogram(
    dist_processed_synthetics,
    x='log_bin',
    y='frequency',
    nbins=40,
    title="Processed Synthetics Distributions Counts",
)

count_dist_measurements = px.histogram(
    dist_measurements,
    x='log_bin',
    y='frequency',
    nbins=40,
    title="Measurements Distributions Counts",
)

sidebar = dbc.Col(
    [
    ],
    style={
        "position": "fixed",
        "top": 72,
        "left": 0,
        "bottom": 0,
        "width": "350px",
        "padding": "20px",
        "backgroundColor": "#3B4F63",
        "overflowY": "auto",
    },
)

content = dbc.Col(
    [
        dbc.Row([
            dbc.Toast(
                [
                    html.P([
                        f"{relativ_processed_measurements.values[0]}%",
                        html.Br(),
                        f"Measurements: {basic_statistics_measurements['total_measurements'].values[0]}",
                        html.Br(),
                        f"Processed Measurements: {basic_statistics_processed_measurements['total_measurements'].values[0]}",
                        html.Br(),
                        f"Number of Peaks: {basic_statistics_processed_measurements["peaks_detected"].values[0]}",
                    ], className="mb-0")
                ],
                header="Processed Measurements",
            ),
            dbc.Toast(
                [
                    html.P([
                        "100%",
                        html.Br(),
                        f"Processed Synthetics: {basic_statistics_processed_synthetics['total_measurements'].values[0]}",
                        html.Br(),
                        f"Number of Peaks: {basic_statistics_processed_synthetics['peaks_detected'].values[0]}",
                    ], className="mb-0")
                ],
                header="Processed Synthetics",
            ),
        ]),
        html.H5(" "),
        html.H5("Statistics"),
        dbc.Row(dash_table.DataTable(data=std_mean_min_max_statistics.to_dict('records'))),
        dbc.Row(dcc.Graph(figure=isotopes_dist_processed_fig)),
        dbc.Row(dcc.Graph(figure=isotopes_dist_synthetics_fig)),
        dbc.Row(dcc.Graph(figure=count_dist_processed_measurements)),
        dbc.Row(dcc.Graph(figure=count_dist_processed_synthetics)),
        dbc.Row(dcc.Graph(figure=count_dist_measurements)),
    ],
    style={
        "marginLeft": "370px",
        "padding": "20px",
        "height": "100vh",
    },
)

layout = dbc.Container(
    dbc.Row([sidebar, content]),
    fluid=True,
    style={"padding": 0}
)

