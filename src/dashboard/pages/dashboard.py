from dash import dcc, html, Input, Output, callback, register_page
import plotly.graph_objects as go
import src.peaks.api as ppi
import src.vae.api as spi
from datetime import datetime
import dash_bootstrap_components as dbc
import plotly.express as px

register_page(__name__, "/", title="RawData")

sidebar = dbc.Col(
    [
        html.H6("Select a processed Measurement", style={"color": "lightgrey"}),
        dcc.Dropdown(
            id="checklist",
            options=[{"label": f, "value": f} for f in ppi.API().unique_dates()],
            value=ppi.API().unique_dates()[0],
            multi=False,
        ),
        html.Br(),
        html.H6("Select a processed Synthetic-Data", style={"color": "lightgrey"}),
        dcc.Dropdown(
            id="checklist_synthetics",
            options=[{"label": f, "value": f} for f in spi.API().re_unique_dates()],
            value=spi.API().re_unique_dates()[0],
            multi=False,
        ),
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
        dbc.Row(dcc.Graph(id="graph2")),
        dbc.Row([dcc.Graph(id="graph3"), html.Div(id="output")]),
    ],
    style={
        "marginLeft": "370px",  # slightly more than sidebar width
        "padding": "20px",
        "height": "100vh",
    },
)

layout = dbc.Container(dbc.Row([sidebar, content]), fluid=True, style={"padding": 0})


@callback(Output("output", "children"), Input("checklist", "value"))
def convert_to_datetime(selected_value):
    if selected_value:
        selected_datetime = datetime.fromisoformat(selected_value)
        return f"Selected DateTime: {type(selected_datetime)}, {selected_datetime}"
    return "Select a date"


@callback(
    Output("graph2", "figure"),  # Output data for the table
    Input("checklist", "value"),
)
def update_combined_chart(file_id):
    fig = go.Figure()
    if file_id:
        file_data = ppi.API().re_measurement(dates=[datetime.fromisoformat(file_id)])
        file_data = file_data.sort_values(by="energy")
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


@callback(
    Output("graph3", "figure"),  # Output data for the table
    Input("checklist_synthetics", "value"),
)
def update_combined_chart_synt(file_id):
    fig = go.Figure()
    try:
        if file_id:
            file_data = spi.API().re_synhtetics(dates=[file_id])
            file_data = file_data.sort_values(by="energy")
            fig.add_trace(
                go.Scatter(
                    x=file_data["energy"],
                    y=file_data["count"],
                    mode="lines",
                    name=f"{file_id}",
                    zorder=10,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=file_data["energy"],
                    y=file_data["background"],
                    mode="lines",
                    name=f"{file_id} - Background",
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
                    name=peak[1],
                    showlegend=True,
                    annotation_text=peak[1],
                )
        fig.update_layout(
            title="Processed Synthetic-Data",
            xaxis_title="Energy",
            yaxis_title="Count / Intensity",
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(type="linear")
    except:
        pass
    return fig
