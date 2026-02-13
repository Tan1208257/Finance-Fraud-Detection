from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Finance Fraud Detection Dashboard", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Parse time columns if present
    for col in ["ts", "hour", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def latest_prediction_file(pred_dir: Path) -> Path | None:
    files = sorted(pred_dir.glob("predictions_*.csv"))
    return files[-1] if files else None


def compute_tick_flags(ticks: pd.DataFrame, spike_pct: float, z_thresh: float) -> pd.DataFrame:
    """
    Adds:
    - dup_event_id_flag
    - missing_any_flag
    - price_spike_flag (per symbol, pct change vs prev tick)
    - price_zscore, outlier_flag (per symbol z-score)
    """
    df = ticks.copy()

    # Duplicates (event_id)
    if "event_id" in df.columns:
        df["dup_event_id_flag"] = df.duplicated(subset=["event_id"], keep=False).astype(int)
    else:
        df["dup_event_id_flag"] = 0

    # Missing values flag (any missing in key cols)
    key_cols = [c for c in ["ts", "symbol", "price", "volume"] if c in df.columns]
    if key_cols:
        df["missing_any_flag"] = df[key_cols].isna().any(axis=1).astype(int)
    else:
        df["missing_any_flag"] = 0

    # Ensure sort
    if "ts" in df.columns and "symbol" in df.columns:
        df = df.sort_values(["symbol", "ts"])

    # Price spike: pct change vs previous tick in same symbol
    if "price" in df.columns and "symbol" in df.columns:
        df["price_prev"] = df.groupby("symbol")["price"].shift(1)
        df["price_spike_pct"] = (df["price"] - df["price_prev"]) / df["price_prev"] * 100.0
        df["price_spike_flag"] = (df["price_spike_pct"].abs() >= spike_pct).fillna(False).astype(int)
    else:
        df["price_prev"] = np.nan
        df["price_spike_pct"] = np.nan
        df["price_spike_flag"] = 0

    # Z-score outliers on price (per symbol)
    if "price" in df.columns and "symbol" in df.columns:
        price_mean = df.groupby("symbol")["price"].transform("mean")
        price_std = df.groupby("symbol")["price"].transform("std").replace(0, np.nan)
        df["price_zscore"] = (df["price"] - price_mean) / price_std
        df["outlier_flag"] = (df["price_zscore"].abs() >= z_thresh).fillna(False).astype(int)
    else:
        df["price_zscore"] = np.nan
        df["outlier_flag"] = 0

    return df


def compute_hourly_flags(hourly: pd.DataFrame, z_thresh: float) -> pd.DataFrame:
    """
    Adds:
    - return_1h_calc from avg_price pct change vs previous hour (per symbol)
    - return_zscore, return_outlier_flag
    """
    df = hourly.copy()

    # Make sure hour is datetime
    if "hour" in df.columns:
        df["hour"] = pd.to_datetime(df["hour"], errors="coerce", utc=True)

    # Sort
    if "symbol" in df.columns and "hour" in df.columns:
        df = df.sort_values(["symbol", "hour"])

    # Compute return from avg_price
    if "avg_price" in df.columns and "symbol" in df.columns:
        df["avg_price_prev"] = df.groupby("symbol")["avg_price"].shift(1)
        df["return_1h_calc"] = (df["avg_price"] - df["avg_price_prev"]) / df["avg_price_prev"]
    else:
        df["avg_price_prev"] = np.nan
        df["return_1h_calc"] = np.nan

    # Z-score outliers on returns (per symbol)
    if "return_1h_calc" in df.columns and "symbol" in df.columns:
        r_mean = df.groupby("symbol")["return_1h_calc"].transform("mean")
        r_std = df.groupby("symbol")["return_1h_calc"].transform("std").replace(0, np.nan)
        df["return_zscore"] = (df["return_1h_calc"] - r_mean) / r_std
        df["return_outlier_flag"] = (df["return_zscore"].abs() >= z_thresh).fillna(False).astype(int)
    else:
        df["return_zscore"] = np.nan
        df["return_outlier_flag"] = 0

    return df


def merge_predictions(hourly: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """
    Merge predictions with hourly analytics on symbol + hour.
    """
    dfh = hourly.copy()
    dfp = pred.copy()

    if "hour" in dfh.columns:
        dfh["hour"] = pd.to_datetime(dfh["hour"], errors="coerce", utc=True)
    if "hour" in dfp.columns:
        dfp["hour"] = pd.to_datetime(dfp["hour"], errors="coerce", utc=True)

    # Only keep useful columns from pred if present
    keep_cols = ["symbol", "hour"]
    for c in ["pred_up_next_proba", "pred_up_next"]:
        if c in dfp.columns:
            keep_cols.append(c)
    dfp = dfp[keep_cols].drop_duplicates(subset=["symbol", "hour"])

    if "symbol" in dfh.columns and "hour" in dfh.columns and "symbol" in dfp.columns and "hour" in dfp.columns:
        return dfh.merge(dfp, on=["symbol", "hour"], how="left")
    return dfh


# -----------------------------
# Sidebar (data root + thresholds)
# -----------------------------
st.sidebar.title("Settings")

# IMPORTANT: default should be "data" since your cwd is project root
DATA_ROOT = Path(st.sidebar.text_input("Project data folder", value="data"))

spike_pct = st.sidebar.slider("Spike threshold (% change per tick)", min_value=1, max_value=200, value=30)
z_thresh = st.sidebar.slider("Outlier threshold (|z|)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
risk_thresh = st.sidebar.slider("ML risk threshold (prob)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

# Paths
ticks_path = DATA_ROOT / "clean" / "ticks_clean.csv"
hourly_path = DATA_ROOT / "analytics" / "hourly_analytics.csv"
pred_dir = DATA_ROOT / "predictions"
pred_path = latest_prediction_file(pred_dir)


# -----------------------------
# Load data
# -----------------------------
st.title("📊 Finance Fraud Detection Dashboard")

colA, colB, colC = st.columns(3)

with colA:
    st.write("**Data root**:", str(DATA_ROOT.resolve()))
with colB:
    st.write("**ticks_clean.csv**:", "✅" if ticks_path.exists() else "❌")
with colC:
    st.write("**hourly_analytics.csv**:", "✅" if hourly_path.exists() else "❌")

ticks = None
hourly = None
pred = None

try:
    ticks = safe_read_csv(ticks_path)
    st.success(f"Loaded ticks_clean.csv ({len(ticks)} rows)")
except Exception as e:
    st.warning(f"Could not load ticks_clean.csv: {e}")

try:
    hourly = safe_read_csv(hourly_path)
    st.success(f"Loaded hourly_analytics.csv ({len(hourly)} rows)")
except Exception as e:
    st.warning(f"Could not load hourly_analytics.csv: {e}")

if pred_path is not None:
    try:
        pred = safe_read_csv(pred_path)
        st.success(f"Loaded predictions ({pred_path.name}) ({len(pred)} rows)")
    except Exception as e:
        st.warning(f"Could not load predictions: {e}")
else:
    st.warning("No predictions file found in data/predictions (predictions_*.csv)")


# -----------------------------
# Sidebar filter: symbol
# -----------------------------
symbols = []
if ticks is not None and "symbol" in ticks.columns:
    symbols = sorted(ticks["symbol"].dropna().unique().tolist())
symbol = st.sidebar.selectbox("Symbol", symbols if symbols else ["None"])


def df_for_symbol(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    if symbol == "None":
        return df
    if "symbol" in df.columns:
        return df[df["symbol"] == symbol].copy()
    return df


# -----------------------------
# Compute fraud/anomaly signals
# -----------------------------
tick_flags = None
hourly_flags = None
hourly_plus_pred = None

if ticks is not None:
    tick_flags = compute_tick_flags(ticks, spike_pct=spike_pct, z_thresh=z_thresh)

if hourly is not None:
    hourly_flags = compute_hourly_flags(hourly, z_thresh=z_thresh)

if hourly_flags is not None and pred is not None:
    hourly_plus_pred = merge_predictions(hourly_flags, pred)
else:
    hourly_plus_pred = hourly_flags


# -----------------------------
# FRAUD SIGNALS (KPIs)
# -----------------------------
st.subheader("🚨 Fraud / Anomaly Signals (Rules + ML)")

k1, k2, k3, k4, k5 = st.columns(5)

if tick_flags is not None:
    dup_count = int(tick_flags["dup_event_id_flag"].sum())
    miss_count = int(tick_flags["missing_any_flag"].sum())
    spike_count = int(tick_flags["price_spike_flag"].sum())
    out_count = int(tick_flags["outlier_flag"].sum())
else:
    dup_count = miss_count = spike_count = out_count = 0

if hourly_plus_pred is not None and "pred_up_next_proba" in hourly_plus_pred.columns:
    avg_risk = float(pd.to_numeric(hourly_plus_pred["pred_up_next_proba"], errors="coerce").mean() or 0)
    ml_flag_count = int((pd.to_numeric(hourly_plus_pred["pred_up_next_proba"], errors="coerce") >= risk_thresh).sum())
else:
    avg_risk = 0.0
    ml_flag_count = 0

k1.metric("Duplicate events", dup_count)
k2.metric("Missing rows", miss_count)
k3.metric(f"Spike flags (≥{spike_pct}%)", spike_count)
k4.metric(f"Outliers (|z|≥{z_thresh})", out_count)
k5.metric(f"ML high-risk (≥{risk_thresh})", ml_flag_count)


# -----------------------------
# Charts: Price & Volume over time (hourly)
# -----------------------------
st.markdown("---")
st.subheader("📈 Market Overview")

left, right = st.columns(2)

hourly_sym = df_for_symbol(hourly_plus_pred)

with left:
    st.write("### Price over time (hourly avg)")
    if hourly_sym is not None and {"hour", "avg_price"}.issubset(hourly_sym.columns):
        plot_df = hourly_sym[["hour", "avg_price"]].dropna().sort_values("hour")
        plot_df = plot_df.set_index("hour")
        st.line_chart(plot_df)
    else:
        st.info("No price data available")

with right:
    st.write("### Volume over time (hourly total)")
    if hourly_sym is not None and {"hour", "total_volume"}.issubset(hourly_sym.columns):
        plot_df = hourly_sym[["hour", "total_volume"]].dropna().sort_values("hour")
        plot_df = plot_df.set_index("hour")
        st.line_chart(plot_df)
    else:
        st.info("No volume data available")


# -----------------------------
# Fraud table: hourly (rules + ML)
# -----------------------------
st.markdown("---")
st.subheader("🧾 Fraud-Risk Table (Hourly)")

if hourly_sym is not None:
    cols = []
    for c in [
        "symbol", "hour", "avg_price", "total_volume", "trades",
        "return_1h_calc", "return_zscore", "return_outlier_flag",
        "pred_up_next_proba", "pred_up_next",
    ]:
        if c in hourly_sym.columns:
            cols.append(c)

    view = hourly_sym[cols].copy() if cols else hourly_sym.copy()

    # Create a clean risk flag based on threshold
    if "pred_up_next_proba" in view.columns:
        view["ml_risk_flag"] = (pd.to_numeric(view["pred_up_next_proba"], errors="coerce") >= risk_thresh).astype(int)

    # Show only suspicious rows toggle
    only_suspicious = st.checkbox("Show only suspicious rows (outlier OR ML high-risk)", value=True)

    if only_suspicious:
        mask = pd.Series(False, index=view.index)
        if "return_outlier_flag" in view.columns:
            mask = mask | (view["return_outlier_flag"] == 1)
        if "ml_risk_flag" in view.columns:
            mask = mask | (view["ml_risk_flag"] == 1)
        view = view[mask]

    st.dataframe(view, use_container_width=True)
else:
    st.info("No hourly analytics loaded.")


# -----------------------------
# Fraud table: tick-level (spikes/outliers/dups)
# -----------------------------
st.markdown("---")
st.subheader("🔎 Tick-Level Suspicious Events (Rules)")

tick_sym = df_for_symbol(tick_flags)

if tick_sym is not None:
    cols2 = []
    for c in [
        "event_id", "ts", "symbol", "price", "volume",
        "dup_event_id_flag", "missing_any_flag",
        "price_spike_pct", "price_spike_flag",
        "price_zscore", "outlier_flag",
    ]:
        if c in tick_sym.columns:
            cols2.append(c)

    view2 = tick_sym[cols2].copy() if cols2 else tick_sym.copy()

    only_suspicious_ticks = st.checkbox(
        "Show only suspicious ticks (dup OR missing OR spike OR outlier)",
        value=True,
        key="only_suspicious_ticks",
    )

    if only_suspicious_ticks:
        mask2 = (
            (view2.get("dup_event_id_flag", 0) == 1)
            | (view2.get("missing_any_flag", 0) == 1)
            | (view2.get("price_spike_flag", 0) == 1)
            | (view2.get("outlier_flag", 0) == 1)
        )
        view2 = view2[mask2]

    st.dataframe(view2, use_container_width=True)
else:
    st.info("No ticks loaded.")


st.markdown("---")
st.caption(
    "Fraud detection here is implemented as anomaly/suspicious-behavior detection: "
    "rule-based flags (duplicates, missing, spikes, outliers) + ML risk scoring (pred_up_next_proba)."
)
