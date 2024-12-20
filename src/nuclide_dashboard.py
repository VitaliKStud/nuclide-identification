from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import os
import dash_table

df_measurements = pd.read_csv("data\\combined_measurements.csv", sep=",")
df_nuclides = pd.read_csv("data\\combined_nuclides.csv", sep=",", low_memory=False)
df_nuclides = df_nuclides.loc[df_nuclides["intensity"] > 1]

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Combined Line and Bar Chart'),
    dcc.Graph(id="graph"),
    dcc.Dropdown(
        id="checklist",
        options=[{"label": f, "value": f} for f in os.listdir("data\\measurements")],
        multi=True
    ),
    dcc.Checklist(
        id="checklist2",
        options=[{"label": "Am241", "value": "am241"}, {"label": "Co60", "value": "co60"}],
        inline=True
    ),
    html.H5("Filtered Nuclides Data"),
    dash_table.DataTable(id="table")  # Table for displaying filtered nuclides
])


@app.callback(
    [Output("graph", "figure"),
     Output("table", "data")],  # Output data for the table
    [Input("checklist", "value"),
     Input("checklist2", "value")],
)
def update_combined_chart(file_id, nuclide_id):
    # Initialize the main figure
    fig = go.Figure()

    # Data for the line chart (Measurements)
    if file_id:
        mask = df_measurements.ID_File.isin(file_id)
        filtered_data = df_measurements[mask]
        for file in filtered_data["ID_File"].unique():
            file_data = filtered_data[filtered_data["ID_File"] == file]
            fig.add_trace(
                go.Scatter(
                    x=file_data["Energy"],
                    y=file_data["Count"],
                    mode="lines",
                    name=f"File: {file}"
                )
            )

    # Data for the bar chart (Nuclides)
    filtered_nuclides = None
    if nuclide_id:
        mask = df_nuclides.nuclide_id.isin(nuclide_id)
        filtered_nuclides = df_nuclides[mask][["energy", "intensity", "nuclide_id"]].dropna()
        for nuclide in filtered_nuclides["nuclide_id"].unique():
            nuclide_data = filtered_nuclides[filtered_nuclides["nuclide_id"] == nuclide]
            for i in range(len(nuclide_data)):
                fig.add_trace(
                    go.Scatter(
                        x=[nuclide_data["energy"].iloc[i], nuclide_data["energy"].iloc[i]],
                        # Vertical line at each x position
                        y=[0, nuclide_data["intensity"].iloc[i] + 10000],  # Line from y=0 to y=intensity (with offset)
                        mode='lines',
                        name=f"Nuclide: {nuclide}",
                        line=dict(width=2, color='red')
                        )
                    )


    # Layout adjustment
    fig.update_layout(
        title="Combined Line and Bar Chart",
        xaxis_title="Energy",
        yaxis_title="Count / Intensity",
        barmode="overlay"  # Overlay bars and lines
    )

    # Prepare the table data for filtered nuclides
    table_data = filtered_nuclides.to_dict('records') if filtered_nuclides is not None else []

    return fig, table_data  # Return figure and table data


if __name__ == "__main__":
    app.run_server(debug=True)
