from dash import Dash, html, dcc, Input, Output, State
import requests, json

API_BASE = "http://127.0.0.1:5001"

app = Dash(__name__)
app.layout = html.Div([
    html.H3("Starter UI"),
    dcc.Input(id="txt", placeholder="Type something...", value=""),
    html.Button("Send to /predict", id="btn"),
    html.Pre(id="out", style={"whiteSpace": "pre-wrap"})
])

@app.callback(
    Output("out", "children"),
    Input("btn", "n_clicks"),
    State("txt", "value"),
    prevent_initial_call=True
)
def call_api(_, text):
    try:
        r = requests.post(f"{API_BASE}/predict", json={"sample": text or ""})
        # debug help if anything goes wrong:
        # print("Called:", r.request.method, r.request.url, "status", r.status_code)
        r.raise_for_status()
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return f"API call failed: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True, use_reloader=False)


