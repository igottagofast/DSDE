"""Financial and time-series feature engineering pipeline."""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config.settings import (
    ALL_TARGETS,
    EXPORT_STOCKS_8,
    FORECAST_HORIZON_DAYS,
    IMPORT_STOCKS_8,
    PROCESSED_DATA_DIR,
)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI) using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    start_idx = avg_gain.first_valid_index()
    if start_idx is not None:
        pos = series.index.get_loc(start_idx)
        g_vals = avg_gain.to_numpy(dtype=float, copy=True)
        l_vals = avg_loss.to_numpy(dtype=float, copy=True)
        gain_arr = gain.to_numpy(dtype=float)
        loss_arr = loss.to_numpy(dtype=float)
        for i in range(pos + 1, len(series)):
            g_vals[i] = (g_vals[i - 1] * (period - 1) + gain_arr[i]) / period
            l_vals[i] = (l_vals[i - 1] * (period - 1) + loss_arr[i]) / period
        avg_gain = pd.Series(g_vals, index=series.index)
        avg_loss = pd.Series(l_vals, index=series.index)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_yield_slope(df_long: pd.DataFrame, df_short: pd.DataFrame) -> pd.DataFrame:
    """Compute Thai bond yield curve slope (TH_20Y - TH_10Y)."""
    long_df = df_long[["Date", "Close"]].rename(columns={"Close": "Yield_20Y"}).copy()
    short_df = df_short[["Date", "Close"]].rename(columns={"Close": "Yield_10Y"}).copy()
    merged = pd.merge(long_df, short_df, on="Date", how="inner")
    merged["Yield_Curve_Slope"] = merged["Yield_20Y"] - merged["Yield_10Y"]
    return merged[["Date", "Yield_Curve_Slope"]].sort_values("Date").reset_index(drop=True)


def _load_intermediate_datasets(intermediate_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all cleaned df_*.parquet datasets from intermediate_dir."""
    datasets: Dict[str, pd.DataFrame] = {}
    for parquet_path in sorted(intermediate_dir.glob("df_*.parquet")):
        name = parquet_path.stem.removeprefix("df_")
        df = pd.read_parquet(parquet_path)
        df["Date"] = pd.to_datetime(df["Date"])
        datasets[name] = df.sort_values("Date").reset_index(drop=True)
    if not datasets:
        raise FileNotFoundError(f"No cleaned df_*.parquet files found in {intermediate_dir}")
    return datasets


def _require_datasets(datasets: Dict[str, pd.DataFrame], names: List[str]) -> None:
    missing = [name for name in names if name not in datasets]
    if missing:
        raise KeyError(f"Missing required cleaned datasets for feature engineering: {missing}")


def _build_baseline_dataframe(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble baseline macro features and daily stock/portfolio returns."""
    required = (
        IMPORT_STOCKS_8
        + EXPORT_STOCKS_8
        + ["Brent", "US_10Y", "USD_THB", "TH_10Y", "TH_20Y"]
    )
    _require_datasets(datasets, required)

    base_dates = pd.DataFrame({"Date": sorted(datasets["USD_THB"]["Date"].unique())})

    # Daily average returns for import and export portfolios
    import_ret_df = base_dates.copy()
    for sym in IMPORT_STOCKS_8:
        import_ret_df = pd.merge(
            import_ret_df,
            datasets[sym][["Date", "Return"]].rename(columns={"Return": sym}),
            on="Date",
            how="left",
        )
    import_ret_df["Import_Avg"] = import_ret_df[IMPORT_STOCKS_8].mean(axis=1)

    export_ret_df = base_dates.copy()
    for sym in EXPORT_STOCKS_8:
        export_ret_df = pd.merge(
            export_ret_df,
            datasets[sym][["Date", "Return"]].rename(columns={"Return": sym}),
            on="Date",
            how="left",
        )
    export_ret_df["Export_Avg"] = export_ret_df[EXPORT_STOCKS_8].mean(axis=1)

    # Cross-universe mean intraday Volatility_to_Open = (High - Low) / Open
    vol_df = base_dates.copy()
    vol_cols = []
    for sym in IMPORT_STOCKS_8 + EXPORT_STOCKS_8:
        df_stk = datasets[sym][["Date", "High", "Low", "Open"]].copy()
        df_stk[f"Vol_{sym}"] = (df_stk["High"] - df_stk["Low"]) / df_stk["Open"].replace(0, np.nan)
        vol_df = pd.merge(vol_df, df_stk[["Date", f"Vol_{sym}"]], on="Date", how="left")
        vol_cols.append(f"Vol_{sym}")
    vol_df["Volatility_to_Open"] = vol_df[vol_cols].mean(axis=1)

    # Yield curve slope and macro daily returns
    slope_df = calculate_yield_slope(datasets["TH_20Y"], datasets["TH_10Y"])
    brent_df = datasets["Brent"][["Date", "Return"]].rename(columns={"Return": "Brent"})
    us10y_df = datasets["US_10Y"][["Date", "Return"]].rename(columns={"Return": "US_10Y"})
    usdthb_df = datasets["USD_THB"][["Date", "Return"]].rename(columns={"Return": "USD_THB"})

    baseline = base_dates.copy()
    for part in (
        brent_df,
        us10y_df,
        usdthb_df,
        vol_df[["Date", "Volatility_to_Open"]],
        slope_df,
        import_ret_df[["Date", "Import_Avg"]],
        export_ret_df[["Date", "Export_Avg"]],
        datasets["BH"][["Date", "Return"]].rename(columns={"Return": "BH_Pct"}),
        datasets["BDMS"][["Date", "Return"]].rename(columns={"Return": "BDMS_Pct"}),
        datasets["EGCO"][["Date", "Return"]].rename(columns={"Return": "EGCO_Pct"}),
        datasets["BCP"][["Date", "Return"]].rename(columns={"Return": "BCP_Pct"}),
    ):
        baseline = pd.merge(baseline, part, on="Date", how="inner")

    return baseline.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)


def _build_expanded_market_features(
    baseline_df: pd.DataFrame,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create strictly causal (<= t) momentum, RSI, SMA distance, and lagged features."""
    market_assets = ["SET50", "SP500", "SOX", "VIX", "DXY", "USD_THB", "CNY_THB", "JPY_THB"]
    _require_datasets(datasets, market_assets)

    aligned = baseline_df[["Date"]].copy()
    for asset in market_assets:
        asset_close = (
            datasets[asset][["Date", "Close"]]
            .dropna(subset=["Close"])
            .drop_duplicates(subset=["Date"])
            .rename(columns={"Close": f"{asset}_Close"})
        )
        aligned = pd.merge(aligned, asset_close, on="Date", how="left")

    aligned = aligned.sort_values("Date").reset_index(drop=True)
    close_cols = [f"{asset}_Close" for asset in market_assets]
    aligned[close_cols] = aligned[close_cols].ffill().bfill()

    features = pd.DataFrame({"Date": aligned["Date"]})
    for asset in ("SET50", "SOX", "SP500"):
        col = f"{asset}_Close"
        features[f"{asset}_Return_3d"] = aligned[col].pct_change(3)
        features[f"{asset}_Return_5d"] = aligned[col].pct_change(5)

    set50_close = aligned["SET50_Close"]
    sma20 = set50_close.rolling(window=20).mean()
    features["SET50_Distance_SMA20"] = (set50_close - sma20) / sma20.replace(0, np.nan)
    features["SET50_RSI_14"] = calculate_rsi(set50_close, period=14)
    features["SET50_Volatility_Roll5"] = set50_close.pct_change().rolling(window=5).std()

    lag_assets = ("SP500", "SOX", "VIX", "DXY", "USD_THB", "CNY_THB", "JPY_THB")
    for asset in lag_assets:
        daily_ret = aligned[f"{asset}_Close"].pct_change()
        features[f"{asset}_Lag1"] = daily_ret.shift(1)
        features[f"{asset}_Lag2"] = daily_ret.shift(2)

    return features


def run_feature_pipeline(intermediate_dir: str, run_date: str) -> dict:
    """Build baseline + expanded features aligned with 5-day forward targets."""
    inter_path = Path(intermediate_dir)
    datasets = _load_intermediate_datasets(inter_path)

    baseline_df = _build_baseline_dataframe(datasets)
    expanded_df = _build_expanded_market_features(baseline_df, datasets)

    base_feature_cols = ["Brent", "US_10Y", "USD_THB", "Volatility_to_Open", "Yield_Curve_Slope"]
    df_features = pd.merge(
        baseline_df[["Date"] + base_feature_cols],
        expanded_df,
        on="Date",
        how="inner",
    )

    # 5-day forward cumulative return targets
    df_targets = baseline_df[ALL_TARGETS].rolling(window=FORECAST_HORIZON_DAYS).sum().shift(
        -FORECAST_HORIZON_DAYS
    )
    df_targets["Date"] = baseline_df["Date"]

    df_final = pd.merge(df_features, df_targets, on="Date", how="inner")
    df_final = (
        df_final.sort_values("Date")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    if df_final.empty:
        raise ValueError("Feature dataset is empty after alignment and NaN removal.")

    feature_cols = [c for c in df_final.columns if c not in (["Date"] + ALL_TARGETS)]

    out_dir = PROCESSED_DATA_DIR / run_date
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "features_dataset.parquet"
    csv_path = out_dir / "features_dataset.csv"

    df_final.to_parquet(features_path, index=False)
    df_final.to_csv(csv_path, index=False)

    return {
        "run_date": run_date,
        "features_path": str(features_path),
        "csv_path": str(csv_path),
        "rows": int(len(df_final)),
        "feature_cols": feature_cols,
        "target_cols": list(ALL_TARGETS),
    }
