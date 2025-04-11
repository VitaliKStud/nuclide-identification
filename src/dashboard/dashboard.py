from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import dash_table
import src.measurements.api as mpi
import src.nuclide.api as npi
from datetime import datetime
from src.peaks.finder import PeakFinder

# Initialize Dash app
app = Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H4('Combined Line and Bar Chart'),
    dcc.Graph(id="graph"),
    dcc.Dropdown(
        id="checklist",
        options=[{"label": f, "value": f} for f in mpi.unique_dates()],
        multi=False
    ),
    html.H5("Filtered Nuclides Data"),
    html.Div(id="output"),
    dash_table.DataTable(id="table")  # Table for displaying filtered nuclides
])


@app.callback(
    Output("output", "children"),
    Input("checklist", "value")
)
def convert_to_datetime(selected_value):
    if selected_value:
        selected_datetime = datetime.fromisoformat(selected_value)
        return f"Selected DateTime: {type(selected_datetime)}"
    return "Select a date"

# Callback to update chart and table based on user input
@app.callback(
    [Output("graph", "figure"),
     Output("table", "data")],  # Output data for the table
    [Input("checklist", "value")],
)
def update_combined_chart(file_id):
    # Initialize the main figure
    3
    fig = go.Figure()
    # Data for the line chart (Measurements)
    if file_id:

        file_data = mpi.measurement(dates=[datetime.fromisoformat(file_id)])
        file_data = file_data.sort_values(by="energy")
        peaks = PeakFinder().find_possible_nuclides(data=file_data, prominence=500)
        fig.add_trace(
            go.Scatter(
                x=file_data["energy"],
                y=file_data["count"],
                mode="lines",
                name=f"File: {file_id}",
                zorder=10
            )
        )

        for nuclide in peaks["nuclide_id"].unique():
            filtered_nuclide = peaks[peaks["nuclide_id"] == nuclide]
            fig.add_trace(
                go.Scatter(
                    x=[filtered_nuclide["energy"].iloc[0], filtered_nuclide["energy"].iloc[0]],  # Vertical line at each x position
                    y=[0, 100000],  # Line from y=0 to y=intensity (with offset)
                    mode='lines',
                    name=f"Nuclide: {nuclide}",
                    line=dict(width=2, color='red'),
                    zorder=0
                )
            )
        table_data = peaks.to_dict('records') if peaks is not None else []

    # Layout adjustment
    fig.update_layout(
        title="Combined Line and Bar Chart",
        xaxis_title="Energy",
        yaxis_title="Count / Intensity",
        barmode="overlay",
    )
    fig.update_yaxes(type="linear")  # Logarithmic scale for the y-axis

    # Prepare the table data for filtered nuclides
    # table_data = filtered_nuclides.to_dict('records') if filtered_nuclides is not None else []

    return fig, table_data  # Return figure and table data

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
