from dash import dcc, html, Input, Output, callback, register_page
import plotly.graph_objects as go
import src.measurements.api as mpi
import src.peaks.api as ppi
import src.vae.api as spi
from datetime import datetime

register_page(__name__, "/", title="Dashboard")

# Define the layout of the app
layout = html.Div(
    [
        html.H4("Combined Lines and Bara Chart"),
        dcc.Dropdown(
            id="checklist",
            options=[{"label": f, "value": f} for f in ppi.API().unique_dates()],
            value=ppi.API().unique_dates()[0],
            multi=False,
        ),
        dcc.Dropdown(
            id="checklist_synthetics",
            options=[{"label": f, "value": f} for f in spi.API().unique_dates()],
            value=spi.API().unique_dates()[0],
            multi=False,
        ),
        dcc.Graph(id="graph"),
        dcc.Graph(id="graph2"),
        dcc.Graph(id="graph3"),
        html.H5("Filtered Nuclides Data"),
        html.Div(id="output"),
        # dash_table.DataTable(id="table"),  # Table for displaying filtered nuclides
    ]
)


@callback(Output("output", "children"), Input("checklist", "value"))
def convert_to_datetime(selected_value):
    if selected_value:
        selected_datetime = datetime.fromisoformat(selected_value)
        return f"Selected DateTime: {type(selected_datetime)}, {selected_datetime}"
    return "Select a date"


# Callback to update chart and table based on user input
@callback(
    Output("graph", "figure"),  # Output data for the table
    Input("checklist", "value"),
)
def update_combined_chart(file_id):
    fig = go.Figure()
    if file_id:
        file_data = mpi.API().measurement(dates=[datetime.fromisoformat(file_id)])
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
    fig.update_layout(
        title="Combined Line and Bar Chart",
        xaxis_title="Energy",
        yaxis_title="Count / Intensity",
        barmode="overlay",
    )
    fig.update_yaxes(type="linear")
    return fig


@callback(
    Output("graph2", "figure"),  # Output data for the table
    Input("checklist", "value"),
)
def update_combined_chart(file_id):
    fig = go.Figure()
    if file_id:
        file_data = ppi.API().measurement(dates=[datetime.fromisoformat(file_id)])
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
        energy_peaks = file_data.loc[file_data["peak"] == 1][
            ["energy", "identified_isotope"]
        ].to_numpy()
        for peak in energy_peaks:
            fig.add_vline(
                x=peak[0],
                line_width=3,
                line_dash="dash",
                line_color="green",
                name=peak[1],
                annotation_text=peak[1],
            )
    fig.update_layout(
        title="Combined Line and Bar Chart",
        xaxis_title="Energy",
        yaxis_title="Count / Intensity",
        barmode="overlay",
    )
    fig.update_yaxes(type="linear")
    return fig


@callback(
    Output("graph3", "figure"),  # Output data for the table
    Input("checklist_synthetics", "value"),
)
def update_combined_chart_synt(file_id):
    fig = go.Figure()
    if file_id:
        file_data = spi.API().synthetic(dates=[file_id])
        file_data = file_data.sort_values(by="energy")
        fig.add_trace(
            go.Scatter(
                x=file_data["energy"],
                y=file_data["count"],
                mode="lines",
                name=f"File: {file_id}",
                zorder=10,
            )
        )
        energy_peaks = file_data.loc[file_data["peak"] == 1][
            ["energy", "identified_isotope"]
        ].to_numpy()
        for peak in energy_peaks:
            fig.add_vline(
                x=peak[0],
                line_width=3,
                line_dash="dash",
                line_color="green",
                name=peak[1],
                annotation_text=peak[1],
            )
    fig.update_layout(
        title="Combined Line and Bar Chart u22",
        xaxis_title="Energy",
        yaxis_title="Count / Intensity",
        barmode="overlay",
    )
    fig.update_yaxes(type="linear")
    return fig
