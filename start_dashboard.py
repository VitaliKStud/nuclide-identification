from dash import html, dcc, Dash
import dash
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True, pages_folder="src/dashboard/pages", external_stylesheets=[dbc.themes.FLATLY])

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Nuclide-Dashboard", href="/", className="ms-2"),
        dbc.Nav([
            dbc.NavLink(f"{page["title"]}", href=page["relative_path"])
            for page in dash.page_registry.values()
        ], className="ms-auto", navbar=True)
    ]),
    color="primary",
    dark=True
)

app.layout = html.Div(
    [
        navbar,
        html.Div(
            dash.page_container,
            style={
                "overflowY": "auto",
                "flex": "1",
                "padding": "20px"
            },
        )
    ],
    style={
        "display": "flex",
        "flexDirection": "column",
        "height": "100vh",
        "overflow": "hidden"
    }
)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, host="0.0.0.0", port=8050)
