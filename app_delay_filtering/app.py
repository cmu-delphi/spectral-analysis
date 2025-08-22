import numpy as np
import pandas as pd
from typing import Tuple
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import gamma as gamma_dist, gaussian_kde
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import warnings

# more real data for delay disribution
# https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024?utm_source=chatgpt.com
# https://datacatalog.med.nyu.edu/dataset/10438?utm_source=chatgpt.com

warnings.filterwarnings("ignore", category=FutureWarning)

# =============== Load Confirmed Case Data (AR states file) ===============
df_full = pd.read_csv("./data/combined_state_no_revision.csv")
df_full["time_value"] = pd.to_datetime(df_full["time_value"])
df_full = df_full.dropna(subset=["JHU-Cases"])
df_full = df_full.sort_values("time_value")
geo_options = sorted(df_full["geo_value"].unique())

# =============== Helper: Real delay sources =====================
REAL_SOURCES = {
    "cn30": "./data/cn30_linelist_reporting_delay_from_symptom_onset.csv",
    "uscdc": "./data/uscdc_linelist_reporting_delay_from_symptom_onset.csv",
    "hk": "./data/hk_linelist_reporting_delay_from_symptom_onset.csv",
}

def load_real_delay_source(kind: str) -> pd.DataFrame:
    """
    Load a real linelist-aggregated delay file with columns:
      reference_date, report_date, count, delay

    Rules:
      - Always drop negative delays (Gamma support is [0, inf)).
      - For 'uscdc' ONLY, also drop delay == 0 (i.e., keep delay > 0).
      - For other sources, keep delay == 0.
    """
    path = REAL_SOURCES.get(kind)
    if path is None:
        raise ValueError(f"Unknown real-source: {kind}")
    df = pd.read_csv(path)

    # Basic normalization
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["count"] = df["count"].astype(float)
    df["delay"] = df["delay"].astype(int)

    # Always remove negative delays
    df = df[df["delay"] >= 0].copy()

    # Additional rule for US CDC only: remove delay == 0
    if kind == "uscdc":
        df = df[df["delay"] > 0].copy()

    return df

# Cache (simple in-memory) so we don't re-read on every callback
_real_cache = {}

def get_real_df(kind: str) -> pd.DataFrame:
    if kind not in _real_cache:
        _real_cache[kind] = load_real_delay_source(kind)
    return _real_cache[kind]

# =============== Rolling kernel builder (Histogram or KDE) ===============
def build_delay_kernel_for_window(
    df_delays: pd.DataFrame,
    window_end: pd.Timestamp,
    window_days: int = 30,
    max_delay: int = 60,
    method: str = "hist",  # 'hist' or 'kde'
) -> Tuple[np.ndarray, int]:
    """
    Make a discrete delay kernel on grid 0..max_delay for the window (t-Y, t].
    Returns (kernel, case_count_in_window)
    """
    t0 = window_end - pd.Timedelta(days=window_days)
    window = df_delays[(df_delays["reference_date"] > t0) & (df_delays["reference_date"] <= window_end)]
    if window.empty:
        return np.zeros(max_delay + 1, dtype=float), 0

    # Aggregate within window by delay
    agg = window.groupby("delay", as_index=False)["count"].sum()
    agg = agg.sort_values("delay")

    # Limit to [0, max_delay]
    agg = agg[agg["delay"] <= max_delay]
    if agg.empty:
        return np.zeros(max_delay + 1, dtype=float), 0

    grid = np.arange(0, max_delay + 1)

    if method == "hist":
        hist = np.zeros_like(grid, dtype=float)
        delays = agg["delay"].to_numpy()
        counts = agg["count"].to_numpy()
        hist[delays] = counts
        total = hist.sum()
        if total > 0:
            hist = hist / total
        return hist, int(counts.sum())

    # KDE
    kde = gaussian_kde(agg["delay"].to_numpy(), weights=agg["count"].to_numpy())
    pdf = kde(grid)
    pdf = np.clip(pdf, 0, None)
    s = pdf.sum()
    if s > 0:
        pdf = pdf / s
    return pdf, int(agg["count"].sum())

# =============== Method-of-moments Gamma fit from weighted delays =========
def gamma_moments_from_window(df_delays: pd.DataFrame, window_end: pd.Timestamp, window_days: int = 30):
    t0 = window_end - pd.Timedelta(days=window_days)
    window = df_delays[(df_delays["reference_date"] > t0) & (df_delays["reference_date"] <= window_end)]
    if window.empty:
        return np.nan, np.nan, 0, np.nan, np.nan

    agg = window.groupby("delay", as_index=False)["count"].sum()
    w = agg["count"].to_numpy(dtype=float)
    x = agg["delay"].to_numpy(dtype=float)
    N = w.sum()
    if N <= 0:
        return np.nan, np.nan, 0, np.nan, np.nan
    mu = (w * x).sum() / N
    m2 = (w * (x ** 2)).sum() / N
    var = m2 - mu ** 2
    if mu <= 0 or var <= 0:
        return mu, var, int(N), np.nan, np.nan
    k = mu ** 2 / var
    theta = var / mu
    return mu, var, int(N), k, theta

# =============== Synthetic Gamma delay kernel =============================
def get_gamma_delay(length: int, mean: float, scale: float) -> np.ndarray:
    x = np.arange(length)
    a = mean / scale
    delay = gamma_dist.pdf(x, a=a, scale=scale)
    s = delay.sum()
    if s > 0:
        delay = delay / s
    return delay

# =============== Deconvolution Methods ===================================
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
    recon_fft = np.conj(delay_fft) * obs_fft / (power_H + 1.0 / snr)
    recon = np.real(ifft(recon_fft))
    recon[recon < 0] = 0
    return recon

# =============== NEW: Forward model (reconvolution) + RMSE ===============
def reconvolve_linear(signal, kernel, out_len=None):
    """
    Linear (non-circular) convolution of signal with kernel, truncated to out_len.
    Kernel is normalized to sum to 1 so scales match the observed series.
    """
    if out_len is None:
        out_len = len(signal)
    k = np.array(kernel, dtype=float)
    ks = k.sum()
    if ks > 0:
        k = k / ks
    y_full = np.convolve(np.asarray(signal, dtype=float), k, mode="full")
    return y_full[:out_len]

def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

# =============== Dash App UI =============================================
app = Dash(__name__)
app.title = "COVID Delay-Aware Deconvolution (Rolling Kernel)"

app.layout = html.Div([
    html.Div([
        html.H2("Reverse Confirmed Cases to Infection Curve"),
        html.Label("Select Region:"),
        dcc.Dropdown(id="geo-dropdown",
                     options=[{"label": g, "value": g} for g in geo_options],
                     value=geo_options[0]),
        html.Br(),
        html.Label("Delay Source:"),
        dcc.Dropdown(
            id="delay-source",
            options=[
                {"label": "Synthetic (Gamma)", "value": "synthetic"},
                {"label": "Real (30 Provinces in China linelist)", "value": "cn30"},
                {"label": "Real (US CDC linelist)", "value": "uscdc"},
                {"label": "Real (Hong Kong linelist)", "value": "hk"},
            ],
            value="synthetic"
        ),
        html.Div(id="real-source-info", style={"color": "blue", "marginTop": "5px"}),  # <-- Blue info line
        html.Div([
            html.Label("Gamma Mean:"),
            dcc.Input(id="mean", type="number", value=5, step=0.1),
            html.Br(), html.Br(),
            html.Label("Gamma Scale:"),
            dcc.Input(id="scale", type="number", value=1, step=0.1),
        ], id="gamma-controls", style={"marginTop": "8px"}),
        html.Div([
            html.Label("Kernel Method:"),
            dcc.RadioItems(
                id="kernel-method",
                options=[
                    {"label": "Histogram", "value": "hist"},
                    {"label": "KDE (weighted)", "value": "kde"}
                ],
                value="hist",
                labelStyle={"display": "inline-block", "marginRight": "10px"}
            ),
            html.Br(),
            html.Label("Window length (days, Y):"),
            dcc.Input(id="window-days", type="number", value=30, min=1, step=1),
            html.Br(), html.Br(),
            html.Label("Max delay to consider (days):"),
            dcc.Input(id="max-delay", type="number", value=60, min=1, step=1),
            html.Br(), html.Br(),
            html.Label("Window end date (t):"),
            dcc.DatePickerSingle(id="window-end",
                                 display_format="YYYY-MM-DD",
                                 placeholder="YYYY-MM-DD")
        ], id="real-controls", style={"display": "none", "marginTop": "8px"}),
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

# Toggle gamma vs real controls
@app.callback(
    Output("gamma-controls", "style"),
    Output("real-controls", "style"),
    Input("delay-source", "value")
)
def toggle_controls(delay_source):
    if delay_source == "synthetic":
        return {"display": "block"}, {"display": "none"}
    else:
        return {"display": "none"}, {"display": "block"}

# Update info line for real sources
@app.callback(
    Output("real-source-info", "children"),
    Input("delay-source", "value")
)
def update_real_source_info(delay_source):
    if delay_source == "synthetic":
        return ""
    try:
        df_real = get_real_df(delay_source)
        total_count = df_real["count"].sum()
        min_date = df_real["reference_date"].min().date()
        max_date = df_real["reference_date"].max().date()
        return f"Total counts: {int(total_count)}, available window end date: {min_date} to {max_date}"
    except Exception as e:
        return f"Error loading real source: {e}"

@app.callback(
    Output("result-plot", "figure"),
    Output("delay-plot", "figure"),
    Output("fft-plot", "figure"),
    Input("update-btn", "n_clicks"),
    State("geo-dropdown", "value"),
    State("delay-source", "value"),
    State("mean", "value"),
    State("scale", "value"),
    State("kernel-method", "value"),
    State("window-days", "value"),
    State("max-delay", "value"),
    State("window-end", "date"),
)
def update_figures(n_clicks, geo_value, delay_source, mean, scale,
                   kernel_method, window_days, max_delay, window_end_str):
    # --- Prepare observed series ---
    df = df_full[df_full["geo_value"] == geo_value]
    df = df.groupby("time_value")["JHU-Cases"].mean().reset_index().sort_values("time_value")
    confirmed = df["JHU-Cases"].values.astype(float)
    time = df["time_value"].values

    # --- Build/choose delay kernel ---
    if delay_source == "synthetic":
        L = min(len(confirmed), (max_delay if max_delay else 60) + 1)
        delay_kernel = get_gamma_delay(L, mean, scale)
        label = f"Gamma(mean={mean}, scale={scale})"
        mu = np.sum(np.arange(L) * delay_kernel)
        var = np.sum((np.arange(L) ** 2) * delay_kernel) - mu**2
        gamma_fit_txt = f" (μ≈{mu:.2f}, σ²≈{var:.2f})"
        n_cases = None
        date_range_txt = ""
    else:
        real_df = get_real_df(delay_source)
        if window_end_str is None:
            window_end = real_df["reference_date"].max()
        else:
            window_end = pd.to_datetime(window_end_str)
        Y = int(window_days) if window_days else 30
        Dmax = int(max_delay) if max_delay else 60
        delay_kernel, n_cases = build_delay_kernel_for_window(
            real_df, window_end=window_end, window_days=Y, max_delay=Dmax, method=kernel_method
        )
        mu, var, _, k, theta = gamma_moments_from_window(real_df, window_end=window_end, window_days=Y)
        src_label = delay_source.upper()
        date_range_txt = f"available [{real_df['reference_date'].min().date()}, {real_df['reference_date'].max().date()}]"
        if not np.isnan(k):
            gamma_fit_txt = f"  (MoM Gamma: k={k:.2f}, θ={theta:.2f}, μ={mu:.2f}, σ²={var:.2f})"
        else:
            gamma_fit_txt = f"  (μ={mu:.2f}, σ²={var:.2f}, MoM Gamma unavailable)"
        label = f"{src_label} {kernel_method.upper()} — window (t-{Y}, t], t={window_end.date()}, N={n_cases}, {date_range_txt}"

    # Make sure kernel doesn't exceed length of observed
    if len(delay_kernel) > len(confirmed):
        delay_kernel = delay_kernel[:len(confirmed)]
        delay_kernel = delay_kernel / delay_kernel.sum() if delay_kernel.sum() > 0 else delay_kernel

    # --- Deconvolution (reconstructed infections) ---
    fft_curve = fft_deconvolution(confirmed, delay_kernel)
    wiener_curve = wiener_deconvolution(confirmed, delay_kernel)

    # --- Forward-model to observed scale (reconvolution) for comparison ---
    fft_reconv = reconvolve_linear(fft_curve, delay_kernel, out_len=len(confirmed))
    wiener_reconv = reconvolve_linear(wiener_curve, delay_kernel, out_len=len(confirmed))

    # --- Fit diagnostics (RMSE vs. raw confirmed) ---
    rmse_fft = rmse(confirmed, fft_reconv)
    rmse_wiener = rmse(confirmed, wiener_reconv)

    # --- Figure 1: Observed vs reconstructions + reconvolved predictions ---
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=time, y=confirmed, mode='lines', name='Confirmed Cases'))

    # Reconstructed infections
    fig_main.add_trace(go.Scatter(x=time, y=fft_curve, mode='lines', name='FFT deconv'))
    fig_main.add_trace(go.Scatter(x=time, y=wiener_curve, mode='lines', name='Wiener deconv'))

    # Forward-modeled expected observed (reconvolved)
    fig_main.add_trace(go.Scatter(
        x=time, y=fft_reconv, mode='lines',
        line=dict(dash='dash'),
        name=f'FFT re-conv (RMSE {rmse_fft:.1f})'
    ))
    fig_main.add_trace(go.Scatter(
        x=time, y=wiener_reconv, mode='lines',
        line=dict(dash='dash'),
        name=f'Wiener re-conv (RMSE {rmse_wiener:.1f})'
    ))

    fig_main.update_layout(title=f"US Infection Reconstruction in {geo_value.upper()}",
                           xaxis_title="Date", yaxis_title="Cases", height=350,
                           legend=dict(
                               orientation="h",
                               yanchor="bottom",
                               y=1.05,
                               xanchor="center",
                               x=0.5,
                               traceorder="normal",
                               tracegroupgap=0  # no extra vertical spacing
                           ),
                           margin=dict(t=100)  # extra top margin for title + legend
    )

    # --- Figure 2: Delay kernel ---
    grid = np.arange(len(delay_kernel))
    fig_delay = go.Figure()
    fig_delay.add_trace(go.Scatter(x=grid, y=delay_kernel, mode='lines', name="Delay kernel"))
    fig_delay.update_layout(title=f"Delay Distribution — {label}{gamma_fit_txt}",
                            xaxis_title="Delay (days)", yaxis_title="Probability", height=300)

    # --- Figure 3: FFT power spectrum of delay kernel ---
    delay_fft = np.abs(fft(delay_kernel)) ** 2
    freqs = fftfreq(len(delay_kernel), d=1)
    mask = freqs > 0
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=freqs[mask], y=delay_fft[mask], mode='lines', name="Power"))
    fig_fft.update_layout(title=f"FFT Power Spectrum of Delay Distribution",
                          xaxis_title="Frequency (cycles/day)", yaxis_title="Power", height=300)

    return fig_main, fig_delay, fig_fft

if __name__ == "__main__":
    app.run(debug=False)