from dash import dcc, html, Input, Output, callback, register_page
import plotly.graph_objects as go
import src.measurements.api as mpi
import src.peaks.api as ppi
import src.vae.api as spi
from datetime import datetime
from src.generator.generator import Generator
import numpy as np
import dash_bootstrap_components as dbc


register_page(__name__, "/generator", title="Dashboard")

layout = html.Div(
    [
        html.H4("Combined Lines and Bara Chart"),
        dcc.Graph(id="generator-graph"),
        dbc.Row([dbc.Col(
        [dcc.Slider(-10.0, 10.0, 0.1, value=-3.0, id="generator-slider",
                   tooltip={"placement": "bottom", "always_visible": True},
                   updatemode="drag"),
        dcc.Slider(-10.0, 10.0, 0.1, value=-3.0, id="generator-slider2",
                   tooltip={"placement": "bottom", "always_visible": True},
                   updatemode="drag"),]
        )])
    ]
)

# Callback to update chart and table based on user input
@callback(
    Output("generator-graph", "figure"),  # Output data for the table
    Input("generator-slider", "value"),
)
def update_generator_plot(generator_slider):
    latent_space = []
    for i in range(10):
        data_to_generate = np.arange(-1, 1, 1 / 12, dtype="float32")
        data_to_generate[1] = generator_slider + i
        latent_space.append(data_to_generate)

    generator = Generator()
    gen = generator.generate(latent_space=latent_space)

    fig = go.Figure()
    for energy_axis, generated_data in gen:
        fig.add_trace(
            go.Scatter(
                x=energy_axis,
                y=generated_data,
                mode="lines",
                zorder=10,
            )
        )
    return fig