import numpy as np
import pandas as pd
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import gamma, gaussian_kde
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

# === Load Real Delay Data ===
# delay_df = pd.read_csv("./spectral-analysis/data/[cdc_line_list]reporting_delay_from_symptom_onset_to_report.csv",
#                        parse_dates=["reference_date", "report_date"])

delay_df = pd.read_csv("./data/[cdc_line_list]reporting_delay_from_symptom_onset_to_report.csv",
                       parse_dates=["reference_date", "report_date"])
# delay_df["cdc_case_earliest_dt"] = pd.to_datetime(delay_df["cdc_case_earliest_dt "])
# delay_df["cdc_report_dt"] = pd.to_datetime(delay_df["cdc_report_dt"])
# delay_df["delay"] = (delay_df["cdc_report_dt"] - delay_df["cdc_case_earliest_dt"]).dt.days
delay_df = delay_df.dropna(subset=["delay"])
delay_df = delay_df[delay_df["delay"] >= 0]

# --- Group by report month ---
delay_df["month"] = delay_df["report_date"].dt.to_period("M").astype(str)
months_available = sorted(delay_df["month"].unique())

# Precompute delay kernels by month
delay_kernels_hist = {}
delay_kernels_kde = {}

for month in months_available:
    sub_df = delay_df[delay_df["month"] == month]

    # Reconstruct full sample of delays by repeating each delay value by its count
    delays_expanded = np.repeat(sub_df["delay"].values, sub_df["count"].astype(int).values)

    if len(delays_expanded) < 2:
        continue

    # Histogram
    hist, _ = np.histogram(delays_expanded, bins=np.arange(0, 61), density=False)
    hist = hist / hist.sum()
    delay_kernels_hist[month] = hist

    # KDE
    try:
        kde = gaussian_kde(delays_expanded)
        xs = np.arange(0, 61)
        pdf = kde(xs)
        pdf = pdf / pdf.sum()
        delay_kernels_kde[month] = pdf
    except Exception:
        continue

# === Load Confirmed Case Data ===
df_full = pd.read_csv("./data/ar_state.csv")
df_full["time_value"] = pd.to_datetime(df_full["time_value"])
df_full = df_full.dropna(subset=["JHU"])
df_full = df_full.sort_values("time_value")
geo_options = sorted(df_full["geo_value"].unique())

# === Deconvolution Functions ===
def get_gamma_delay(length, mean, scale):
    x = np.arange(length)
    a = mean / scale
    delay = gamma.pdf(x, a=a, scale=scale)
    return delay / delay.sum()

def fft_deconvolution(observed, delay, eps=1e-3):
    padded_delay = np.zeros(len(observed))
    padded_delay[:len(delay)] = delay
    obs_fft = fft(observed)
    delay_fft = fft(padded_delay)
    delay_fft[np.abs(delay_fft) < eps] = eps
    recon_fft = obs_fft / delay_fft
    recon = np.real(ifft(recon_fft))
    recon[recon < 0] = 0
    return recon

def wiener_deconvolution(observed, delay, snr=10):
    padded_delay = np.zeros(len(observed))
    padded_delay[:len(delay)] = delay
    obs_fft = fft(observed)
    delay_fft = fft(padded_delay)
    power_H = np.abs(delay_fft) ** 2
    recon_fft = np.conj(delay_fft) * obs_fft / (power_H + 1 / snr)
    recon = np.real(ifft(recon_fft))
    recon[recon < 0] = 0
    return recon

# === App Layout ===
app = Dash(__name__)
app.title = "COVID Delay-Aware Deconvolution"

app.layout = html.Div([
    html.Div([
        html.H2("Reverse Confirmed Cases to Infection Curve"),
        html.Label("Select Region:"),
        dcc.Dropdown(id="geo-dropdown",
                     options=[{"label": g, "value": g} for g in geo_options],
                     value=geo_options[0]),
        html.Br(),
        html.Label("Delay Source:"),
        dcc.Dropdown(id="delay-source",
                     options=[
                         {"label": "Synthetic (Gamma)", "value": "synthetic"},
                         {"label": "Real (CDC, Histogram)", "value": "hist"},
                         {"label": "Real (CDC, KDE)", "value": "kde"},
                     ],
                     value="synthetic"),
        html.Br(),
        html.Div([
            html.Label("Gamma Mean:"),
            dcc.Input(id="mean", type="number", value=5, step=0.1),
            html.Br(), html.Br(),
            html.Label("Gamma Scale:"),
            dcc.Input(id="scale", type="number", value=1, step=0.1),
        ], id="gamma-controls"),

        html.Div([
            html.Label("Delay Month (for Display Only):"),
            dcc.Dropdown(id="week-selector",
                         options=[{"label": m, "value": m} for m in months_available],
                         value=months_available[0])
        ], id="week-controls", style={"display": "none"}),

        html.Br(),
        html.Button("Update", id="update-btn", n_clicks=0)
    ], style={"width": "25%", "padding": "20px"}),

    html.Div([
        dcc.Graph(id="result-plot", style={"height": "350px"}),
        html.Div([
            dcc.Graph(id="delay-plot", style={"width": "49%", "height": "300px", "display": "inline-block"}),
            dcc.Graph(id="fft-plot", style={"width": "49%", "height": "300px", "display": "inline-block"})
        ])
    ], style={"width": "75%", "padding": "20px"})
], style={"display": "flex", "flexDirection": "row", "flexWrap": "nowrap", "width": "100vw"})

# === Toggle Controls Based on Delay Source ===
@app.callback(
    Output("gamma-controls", "style"),
    Output("week-controls", "style"),
    Input("delay-source", "value")
)
def toggle_controls(delay_source):
    if delay_source == "synthetic":
        return {"display": "block"}, {"display": "none"}
    else:
        return {"display": "none"}, {"display": "block"}

# === Update Figures ===
@app.callback(
    Output("result-plot", "figure"),
    Output("delay-plot", "figure"),
    Output("fft-plot", "figure"),
    Input("update-btn", "n_clicks"),
    State("geo-dropdown", "value"),
    State("delay-source", "value"),
    State("mean", "value"),
    State("scale", "value"),
    State("week-selector", "value")
)
def update_figures(n_clicks, geo_value, delay_source, mean, scale, display_month):
    df = df_full[df_full["geo_value"] == geo_value]
    df = df.groupby("time_value")["JHU"].mean().reset_index().sort_values("time_value")
    confirmed = df["JHU"].values
    time = df["time_value"].values

    if delay_source == "synthetic":
        delay = get_gamma_delay(len(confirmed), mean, scale)
        fft_curve = fft_deconvolution(confirmed, delay)
        wiener_curve = wiener_deconvolution(confirmed, delay)
        label = "Gamma PDF"
    else:
        delay = get_gamma_delay(60, 5, 1)  # dummy kernel for Wiener recon
        fft_curve = None
        wiener_curve = wiener_deconvolution(confirmed, delay)
        label = "Empirical Histogram" if delay_source == "hist" else "Smoothed KDE"

    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=time, y=confirmed, mode='lines', name='Confirmed Cases'))
    if fft_curve is not None:
        fig_main.add_trace(go.Scatter(x=time, y=fft_curve, mode='lines', name='FFT Deconvolution'))
    fig_main.add_trace(go.Scatter(x=time, y=wiener_curve, mode='lines', name='Wiener Deconvolution'))
    fig_main.update_layout(
        title=f"Infection Reconstruction in {geo_value.upper()}",
        xaxis_title="Date", yaxis_title="Cases", height=350
    )

    # --- Delay Plot ---
    if delay_source == "hist":
        delay = delay_kernels_hist.get(display_month)
    elif delay_source == "kde":
        delay = delay_kernels_kde.get(display_month)
    else:
        delay = get_gamma_delay(60, mean, scale)

    if delay is None:
        delay = np.zeros(60)

    month_display = display_month.replace("-", "/") if display_month else "N/A"
    fig_delay = go.Figure()
    fig_delay.add_trace(go.Scatter(x=np.arange(len(delay)), y=delay, mode='lines', name=label))
    fig_delay.update_layout(
        title=f"Delay Distribution ({label}) in {month_display}",
        xaxis_title="Days", yaxis_title="Probability", height=300
    )

    delay_fft = np.abs(fft(delay)) ** 2
    freqs = fftfreq(len(delay), d=1)
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=freqs[freqs > 0], y=delay_fft[freqs > 0], mode='lines', name="Power"))
    fig_fft.update_layout(
        title=f"FFT Power Spectrum of Delay Distribution ({label})",
        xaxis_title="Frequency", yaxis_title="Power", height=300
    )

    return fig_main, fig_delay, fig_fft

if __name__ == "__main__":
    app.run_server(debug=False)