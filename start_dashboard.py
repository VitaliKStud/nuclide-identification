from src.dashboard.dashboard import app

app.run_server(debug=True, use_reloader=True, host="127.0.0.1", port=8051)
