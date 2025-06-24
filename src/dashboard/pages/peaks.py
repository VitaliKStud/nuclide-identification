from dash import dcc, html, Input, Output, callback, register_page, dash_table
import plotly.graph_objects as go
import src.measurements.api as mpi
import src.peaks.api as ppi
import src.vae.api as spi
from datetime import datetime
import dash_bootstrap_components as dbc
import plotly.express as px
from src.peaks.finder import PeakFinder
from config.loader import load_config
import src.nuclide.api as npi

register_page(__name__, "/peakfinder", title="PeakFinder")

sidebar = dbc.Col(
    [
        html.H6("Select a Measurement", style={"color": "lightgrey"}),
        dcc.Dropdown(
            id="checklist",
            options=[{"label": f, "value": f} for f in mpi.API().unique_dates()],
            value=mpi.API().unique_dates()[0],
            multi=False,
        ),
        html.Br(),
        html.H6("Nuclides", style={"color": "lightgrey"}),
        dcc.Dropdown(
            id="input-nuclides",
            multi=True,
            options=npi.API().unique_nuclides(),
            value=list(load_config()["peakfinder"]["nuclides"]),
        ),
        html.Br(),
        html.H6("Tolerance", style={"color": "lightgrey"}),
        dcc.Input(
            load_config()["peakfinder"]["tolerance"],
            type="number",
            id="input-tolerance",
        ),
        html.Br(),
        html.H6("Nuclide-Intensity", style={"color": "lightgrey"}),
        dcc.Input(
            load_config()["peakfinder"]["nuclide_intensity"],
            type="number",
            id="input-nuclide-intensity",
        ),
        html.Br(),
        html.H6("Matching-Ration", style={"color": "lightgrey"}),
        dcc.Input(
            load_config()["peakfinder"]["matching_ratio"],
            type="number",
            id="input-matching-ratio",
        ),
        html.Br(),
        html.H6("Prominence", style={"color": "lightgrey"}),
        dcc.Input(
            load_config()["peakfinder"]["prominence"],
            type="number",
            id="input-prominence",
        ),
        html.Br(),
        html.H6("Wlen", style={"color": "lightgrey"}),
        dcc.Input(load_config()["peakfinder"]["wlen"], type="number", id="input-wlen"),
        html.Br(),
        html.H6("Rel-Height", style={"color": "lightgrey"}),
        dcc.Input(
            load_config()["peakfinder"]["rel_height"],
            type="number",
            id="input-relheight",
        ),
        html.Br(),
        html.H6("Width", style={"color": "lightgrey"}),
        dcc.Input(load_config()["peakfinder"]["wlen"], type="number", id="input-width"),
        html.Br(),
        html.H6("---------------", style={"color": "lightgrey"}),
        html.Br(),
        html.H6("Filter Energy Minimum", style={"color": "lightgrey"}),
        dcc.Input(value=0, type="number", id="input-min-energy"),
        html.Br(),
        html.H6("Filter Energy Maximum", style={"color": "lightgrey"}),
        dcc.Input(value=3000, type="number", id="input-max-energy"),
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
        dbc.Row(dcc.Graph(id="graph-peak-finder")),
        dbc.Row(
            [
                dash_table.DataTable(id="table-peak-finder"),
                html.Div(id="posible-nuclides-peak-output"),
            ]
        ),
    ],
    style={
        "marginLeft": "370px",  # slightly more than sidebar width
        "padding": "20px",
        "height": "100vh",
    },
)

layout = dbc.Container(dbc.Row([sidebar, content]), fluid=True, style={"padding": 0})


@callback(
    [
        Output("table-peak-finder", "data"),
        Output("posible-nuclides-peak-output", "children"),
    ],
    [
        Input("input-nuclide-intensity", "value"),
        Input("input-max-energy", "value"),
        Input("input-min-energy", "value"),
    ],
)
def get_nuclide_table(nuclide_intensity, max_energy, min_energy):
    data = (
        npi.API()
        .nuclides_by_intensity_and_min_max_energy(
            intensity=nuclide_intensity, min_energy=min_energy, max_energy=max_energy
        )
        .sort_values(by="energy")
    )
    unique_nuclides = "\n".join(data["nuclide_id"].unique().tolist())
    return data.to_dict("records"), unique_nuclides


@callback(
    Output("graph-peak-finder", "figure"),  # Output data for the table
    [
        Input("checklist", "value"),
        Input("input-tolerance", "value"),
        Input("input-nuclide-intensity", "value"),
        Input("input-matching-ratio", "value"),
        Input("input-prominence", "value"),
        Input("input-wlen", "value"),
        Input("input-nuclides", "value"),
        Input("input-width", "value"),
        Input("input-relheight", "value"),
    ],
)
def update_combined_chart(
    file_id,
    tolerance,
    nuclide_intensity,
    matching_ratio,
    prominence,
    wlen,
    nuclides,
    width,
    rel_height,
):
    fig = go.Figure()
    if file_id:
        interpolate_energy = bool(load_config()["peakfinder"]["interpolate_energy"])

        # nuclides = npi.API().unique_nuclides()
        file_data = PeakFinder(
            selected_date=datetime.fromisoformat(file_id),
            data=mpi.API().measurement([datetime.fromisoformat(file_id)]),
            meta=mpi.API().meta_data([datetime.fromisoformat(file_id)]),
            schema="processed_measurements",
            nuclides=nuclides,
            prominence=prominence,
            tolerance=tolerance,
            wlen=wlen,
            width=width,
            rel_height=rel_height,
            nuclides_intensity=nuclide_intensity,
            matching_ratio=matching_ratio,
            interpolate_energy=interpolate_energy,
        ).process_spectrum(return_detailed_view=True)

        fig.add_trace(
            go.Scatter(
                x=file_data["energy"],
                y=file_data["count"],
                mode="lines",
                name=f"File: {datetime.fromisoformat(file_id)}",
                zorder=10,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=file_data["energy"],
                y=file_data["background"],
                mode="lines",
                name=f"{datetime.fromisoformat(file_id)} - Background",
                zorder=10,
            )
        )
        file_data["cleaned_data"] = file_data["count"] - file_data["background"]
        fig.add_trace(
            go.Scatter(
                x=file_data["energy"],
                y=file_data["cleaned_data"],
                mode="lines",
                name=f"{datetime.fromisoformat(file_id)} - Background",
                zorder=10,
            )
        )
        unique_isotopes = file_data.loc[file_data["peak"] == 1][
            "identified_isotope"
        ].unique()
        color_palette = px.colors.qualitative.Dark24
        isotope_color_map = {
            isotope: color_palette[i % len(color_palette)]
            for i, isotope in enumerate(unique_isotopes)
        }
        energy_peaks = file_data.loc[file_data["peak"] == 1][
            ["energy", "identified_isotope"]
        ].to_numpy()
        for peak in energy_peaks:
            fig.add_vline(
                x=peak[0],
                line_width=2,
                line_dash="dash",
                line_color=isotope_color_map[peak[1]],
                legendgroup=peak[1],
                showlegend=True,
                name=peak[1],
                annotation_text=peak[1],
            )
    fig.update_layout(
        title="Processed Measurement",
        xaxis_title="Energy",
        yaxis_title="Count",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(type="linear")
    return fig
