from dash import dcc, html, Input, Output, callback, register_page
import plotly.graph_objects as go
from src.generator.generator import Generator
import numpy as np
import dash_bootstrap_components as dbc
import random
from config.loader import load_config

latent_dim = load_config()["vae"]["latent_dim"]

register_page(__name__, "/generator", title="Generator")

latent_dim_sliders = [
    dcc.Slider(
        -10.0,
        10.0,
        0.1,
        value=random.uniform(-1, 1),
        id=f"latent-dim:{dim}",
        tooltip={"placement": "bottom", "always_visible": True},
        marks={i: "{}".format(i) for i in range(-10, 11, 2)},
        updatemode="drag",
    )
    for dim in range(latent_dim)
]

sidebar = dbc.Col(
    [
        html.H6("Latent-Dim Parameters", style={"color": "lightgrey"}),
        dbc.Row(latent_dim_sliders),
        html.Br(),
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
        dbc.Row(dcc.Graph(id="generator-graph")),
    ],
    style={
        "marginLeft": "370px",
        "padding": "20px",
        "height": "100vh",
    },
)

layout = dbc.Container(dbc.Row([sidebar, content]), fluid=True, style={"padding": 0})


@callback(
    Output("generator-graph", "figure"),
    [Input(f"latent-dim:{dim}", "value") for dim in range(latent_dim)],
)
def update_generator_plot(*generator_slider):
    latent_space = []
    data_to_generate = np.zeros(latent_dim, dtype="float32")
    for i, latent_value in enumerate(generator_slider):
        data_to_generate[i] = latent_value
    latent_space.append(data_to_generate)

    generator = Generator()
    gen = generator.generate(latent_space=latent_space)

    fig = go.Figure()
    for energy_axis, generated_data, z_latent_space in gen:
        fig.add_trace(
            go.Scatter(
                x=energy_axis,
                y=generated_data,
                mode="lines",
                zorder=10,
            )
        )
    return fig
