"""
Time-Series, Momentum, Lagged, and Technical Indicator Features
"""
import pandas as pd
import numpy as np
from src.data.cleaner import ensure_date_column


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    คำนวณดัชนีกำลังสัมพัทธ์ (Relative Strength Index: RSI) ตามสูตร Wilder's Smoothing
    """
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


def build_base_lags_and_momentum(X_base: pd.DataFrame) -> pd.DataFrame:
    """
    สร้าง Lagged Features, Momentum (% Change), และ Rolling Volatility จากตัวแปรต้นฐาน (X_base)
    """
    X_fe = X_base.copy()

    # 1. Lagged Features
    if 'Brent' in X_fe.columns:
        X_fe['Brent_lag1'] = X_fe['Brent'].shift(1)
        X_fe['Brent_lag2'] = X_fe['Brent'].shift(2)
    if 'USD_THB' in X_fe.columns:
        X_fe['USD_THB_lag1'] = X_fe['USD_THB'].shift(1)
        X_fe['USD_THB_lag2'] = X_fe['USD_THB'].shift(2)
    if 'US_10Y' in X_fe.columns:
        X_fe['US_10Y_lag1'] = X_fe['US_10Y'].shift(1)

    # 2. Percentage Change / Momentum
    if 'USD_THB' in X_fe.columns:
        X_fe['USD_THB_pct_1d'] = X_fe['USD_THB'].pct_change(1).replace([np.inf, -np.inf], np.nan)
        X_fe['USD_THB_pct_5d'] = X_fe['USD_THB'].pct_change(5).replace([np.inf, -np.inf], np.nan)
    if 'Brent' in X_fe.columns:
        X_fe['Brent_pct_5d'] = X_fe['Brent'].pct_change(5).replace([np.inf, -np.inf], np.nan)

    # 3. Rolling Volatility
    if 'Volatility_to_Open' in X_fe.columns:
        X_fe['Volatility_roll5_std'] = X_fe['Volatility_to_Open'].rolling(window=5).std()
    if 'USD_THB' in X_fe.columns:
        X_fe['USD_THB_roll5_std'] = X_fe['USD_THB'].rolling(window=5).std()

    return X_fe


def build_macro_and_market_features(df_daily: pd.DataFrame, index_dict: dict) -> pd.DataFrame:
    """
    รวมดัชนีตลาดโลก (SET50, SP500, SOX, VIX, DXY) และอัตราแลกเปลี่ยน พร้อมคำนวณ RSI, Moving Average, Lags
    """
    df_merged = ensure_date_column(df_daily).copy()

    for name, raw_df in index_dict.items():
        if raw_df is None or raw_df.empty:
            continue
        clean_df = ensure_date_column(raw_df)
        close_col = 'Close' if 'Close' in clean_df.columns else ('ล่าสุด' if 'ล่าสุด' in clean_df.columns else None)
        if close_col is None:
            continue

        series = pd.to_numeric(clean_df[close_col].astype(str).str.replace(',', '', regex=False), errors='coerce')
        temp = pd.DataFrame({'Date': clean_df['Date'], f'{name}_Close': series}).dropna().drop_duplicates(subset=['Date'])
        df_merged = pd.merge(df_merged, temp, on='Date', how='left')

        # Forward fill & Backward fill สำหรับวันหยุดไม่ตรงกัน
        df_merged[f'{name}_Close'] = df_merged[f'{name}_Close'].ffill().bfill()

        # คำนวณ Features เสริม
        df_merged[f'{name}_Return_1d'] = df_merged[f'{name}_Close'].pct_change(1)
        df_merged[f'{name}_Return_5d'] = df_merged[f'{name}_Close'].pct_change(5)
        df_merged[f'{name}_SMA_10'] = df_merged[f'{name}_Close'].rolling(10).mean()
        df_merged[f'{name}_RSI_14'] = calculate_rsi(df_merged[f'{name}_Close'], period=14)

    return df_merged
