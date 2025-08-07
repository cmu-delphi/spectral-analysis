import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === Helper Functions ===
def extract_signal(df, state, signal, date_cutoff="2020-12-31"):
    curve = df.loc[
        (df["geo_value"] == state) & (df["time_value"] <= date_cutoff),
        ["time_value", signal]
    ].sort_values("time_value").dropna()

    if curve.empty:
        return None, None

    t_raw = pd.to_datetime(curve["time_value"])
    y = curve[signal].astype(float).values
    y[np.isnan(y)] = 0
    y -= y.mean()
    return y, t_raw

def preprocess_signal(y, t_raw, pad_length, pad_side="both"):
    if pad_side == "left":
        y_padded = np.concatenate([np.zeros(pad_length), y])
    else:
        y_padded = np.concatenate([np.zeros(pad_length), y, np.zeros(pad_length)])

    full_time = pd.date_range(
        start=t_raw.iloc[0] - pd.to_timedelta(pad_length, unit='D'),
        periods=len(y_padded), freq='D'
    )
    return y_padded, full_time

def compute_fft(y_padded, dt=1.0):
    N = len(y_padded)
    freqs = fftfreq(N, d=dt)
    fft_vals = fft(y_padded)
    return freqs, fft_vals

def apply_frequency_filter(freqs, fft_vals, low_cutoff, high_cutoff, method="hard"):
    fft_filtered = fft_vals.copy()
    if method == "gaussian":
        weights = np.exp(-(freqs / high_cutoff) ** 2)
        fft_filtered *= weights
    else:
        fft_filtered[
            (np.abs(freqs) < low_cutoff)
            | (np.abs(freqs) > high_cutoff)
        ] = 0
    return fft_filtered

def compute_power_spectrum(freqs, fft_vals):
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    magnitude = np.abs(fft_vals[pos_mask])
    periods = 1 / pos_freqs
    return periods, magnitude

def create_fft_plot(full_time, y_padded, filtered_signal,
                    periods, magnitude,
                    signal, state):
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15,
                        subplot_titles=[
                            "Time-Domain Signal (Original vs Filtered)",
                            "FFT Amplitude Spectrum (Period Domain)"
                        ])

    fig.add_trace(go.Scatter(x=full_time, y=y_padded, name="Original", line=dict(color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=full_time, y=filtered_signal, name="Filtered", line=dict(color="blue")), row=1, col=1)

    sorted_idx = np.argsort(periods)
    sorted_periods = periods[sorted_idx]
    sorted_magnitude = magnitude[sorted_idx]
    bar_widths = np.gradient(sorted_periods)

    fig.add_trace(go.Bar(
        x=sorted_periods,
        y=sorted_magnitude,
        width=bar_widths,
        marker=dict(color='rgba(0,128,0,0.4)'),
        name="Amplitude"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=sorted_periods,
        y=sorted_magnitude,
        mode='markers',
        marker=dict(size=4, color='green'),
        showlegend=False
    ), row=2, col=1)

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Period (days)", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", type="log", row=2, col=1)

    fig.update_layout(height=700, title=f"{signal} in {state.upper()} — FFT Analysis", showlegend=False)
    return fig

def generate_fft_figure(df, state, signal, pad_length=10,
                        low_cutoff=0.01, high_cutoff=0.5,
                        filter_type="hard", pad_side="both"):
    y, t_raw = extract_signal(df, state, signal)
    if y is None:
        return go.Figure().update_layout(title="No data available")

    y_padded, full_time = preprocess_signal(y, t_raw, pad_length, pad_side)
    freqs, fft_vals = compute_fft(y_padded)
    filtered_fft_vals = apply_frequency_filter(freqs, fft_vals,
                                               low_cutoff, high_cutoff,
                                               method=filter_type)
    filtered_signal = np.real(ifft(filtered_fft_vals))
    periods, magnitude = compute_power_spectrum(freqs, filtered_fft_vals)

    return create_fft_plot(full_time, y_padded, filtered_signal, periods, magnitude, signal, state)

# === Load Data ===
df = pd.read_csv("./data/ar_state.csv")
available_states = sorted(df["geo_value"].unique())
available_signals = sorted(df.columns.drop(["geo_value", "time_value"]))

# === Dash App ===
app = Dash(__name__)
app.title = "FFT Time Series Explorer"

app.layout = html.Div(style={"display": "flex"}, children=[

    html.Div(style={"width": "25%", "padding": "20px"}, children=[
        html.H3("Controls"),

        html.P([
            "Use the controls below to:",
            html.Br(),
            "• select a region and signal",
            html.Br(),
            "• set the frequency band for filtering (band-pass)",
            html.Br(),
            "• and apply padding to the signal before FFT transformation."
        ], style={"fontSize": "13px", "marginBottom": "20px", "whiteSpace": "normal"}),

        html.Label("Select State:", style={"marginBottom": "5px"}),
        dcc.Dropdown(id="state-dropdown", options=[{"label": s.upper(), "value": s} for s in available_states],
                     value=available_states[0], style={"marginBottom": "20px"}),

        html.Label("Select Signal:", style={"marginBottom": "5px"}),
        dcc.Dropdown(id="signal-dropdown", options=[{"label": s, "value": s} for s in available_signals],
                     value=available_signals[0], style={"marginBottom": "20px"}),

        html.Label("Frequency Band (Low - High):", style={"marginBottom": "5px"}),
        dcc.RangeSlider(id="freq-range", min=0.001, max=0.5, step=0.001, value=[0.001, 0.5],
                        marks={0.01: "0.01", 0.1: "0.1", 0.3: "0.3", 0.5: "0.5"},
                        tooltip={"placement": "bottom"}),  # Add margin below next
        html.Div(style={"marginBottom": "20px"}),  # Spacer

        html.Label("Pad Length (days):  ", style={"marginBottom": "5px"}),
        dcc.Input(id="pad-length", type="number", value=10, min=0, style={"marginBottom": "20px"}),

        html.Label("Pad Side:", style={"marginBottom": "5px"}),
        dcc.Dropdown(id="pad-side",
                     options=[{"label": "Both", "value": "both"},
                              {"label": "Left Only", "value": "left"}],
                     value="both", style={"marginBottom": "20px"})
    ]),

    html.Div(style={"width": "75%", "padding": "20px"}, children=[
        html.P([
            "The upper plot shows the original (gray) and filtered (blue) signal in the time domain.",
            html.Br(),
            "The lower plot shows the amplitude spectrum in the frequency (period) domain after applying FFT.",
            html.Br(),
            "Longer periods (right side) correspond to low-frequency trends."
        ], style={"fontSize": "14px", "marginBottom": "10px"}),

        dcc.Graph(id="fft-figure")
    ])
])

@app.callback(
    Output("fft-figure", "figure"),
    Input("state-dropdown", "value"),
    Input("signal-dropdown", "value"),
    Input("freq-range", "value"),
    Input("pad-length", "value"),
    Input("pad-side", "value")
)
def update_fft_plot(state, signal, freq_range, pad_len, pad_side):
    low_cutoff, high_cutoff = freq_range
    return generate_fft_figure(df, state, signal,
                               pad_length=pad_len,
                               low_cutoff=low_cutoff,
                               high_cutoff=high_cutoff,
                               filter_type="hard",
                               pad_side=pad_side)

if __name__ == "__main__":
    app.run_server(debug=False)